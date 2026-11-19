"""The supervisor that restarts a tool once after a GPU out-of-memory crash.

Real processes stand in for the tool: a small script that calls the supervisor's hook
and then does what its first argument says (exit with a code, write a note and crash, sleep
and report its processes). No GPU and no network are used.
"""

import ast
import contextlib
import io
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

import psutil

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities import Tool_Supervisor  # noqa: E402

WINDOWS = sys.platform == "win32"
OFF = Tool_Supervisor.OFF_VARIABLE
GIB = 1024 ** 3

# The windows code of a crash from C++ (0xC0000409), signed as os._exit takes it.
CRASH_SIGNED = -1073740791
CRASH_UNSIGNED = 0xC0000409
CTRL_C_SIGNED = -1073741510

# A stand-in that dies of a crash signal leaves a crash report in the Mac owner's DiagnosticReports folder.
NO_CRASH_ON_MAC = unittest.skipIf(
    sys.platform == "darwin",
    "a real process ended by a crash signal leaves a crash report on macOS; RestartDecisionTests covers the rule there",
)

STAND_IN = r'''
import json, os, signal, subprocess, sys, time

SRC, MODE, ARGS = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, SRC)
from utilities import Tool_Supervisor
Tool_Supervisor.supervise(__name__, __file__)


def crash():
    if sys.platform == "win32":
        os._exit(-1073740791)
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.kill(os.getpid(), signal.SIGABRT)
    time.sleep(30)


def note():
    Tool_Supervisor.record_note(
        "cuda:0", RuntimeError("CUDA error: out of memory"), where="stand-in",
        free_bytes=11 << 30, total_bytes=12 << 30,
    )


def count(path):
    try:
        with open(path) as handle:
            value = int(handle.read() or 0)
    except OSError:
        value = 0
    with open(path, "w") as handle:
        handle.write(str(value + 1))
    return value + 1


def write_pids(path, **extra):
    pids = {"child": os.getpid(), "parent": Tool_Supervisor._parent_pid, **extra}
    with open(path + ".part", "w") as handle:
        json.dump(pids, handle)
    os.replace(path + ".part", path)


print("start", os.getpid(), flush=True)
if MODE == "exit":
    os._exit(int(ARGS[0]))
elif MODE == "echo":
    print("flags", sys.flags.optimize, "argv", json.dumps(ARGS, ensure_ascii=True), flush=True)
    print("stdin", sys.stdin.read().strip(), flush=True)
    sys.stderr.write("to stderr\n")
elif MODE == "report":
    print("report", json.dumps({
        "note": Tool_Supervisor._note_path,
        "restart": Tool_Supervisor._restart_info,
        "markers": [k for k in os.environ if k.startswith("SSN_TOOL_SUPERVISOR")],
    }), flush=True)
elif MODE == "nested":
    inner = subprocess.run([sys.executable, __file__, SRC, "report"], capture_output=True, text=True)
    print("inner", inner.stdout.strip().splitlines()[-1], flush=True)
    print("outer", Tool_Supervisor._note_path, flush=True)
elif MODE == "crash_once":
    if count(ARGS[0]) == 1:
        note()
        crash()
    print("second run", json.dumps(Tool_Supervisor._restart_info), flush=True)
elif MODE == "crash_always":
    count(ARGS[0])
    note()
    crash()
elif MODE == "note_exit1":
    count(ARGS[0])
    note()
    os._exit(1)
elif MODE == "crash_no_note":
    count(ARGS[0])
    crash()
elif MODE == "note_ctrl_c":
    count(ARGS[0])
    note()
    if sys.platform == "win32":
        os._exit(-1073741510)
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(30)
elif MODE == "sleep":
    write_pids(ARGS[0])
    time.sleep(120)
elif MODE == "sleep_grand":
    grand = subprocess.Popen([sys.executable, "-c", "import os,time; print(os.getpid(), flush=True); time.sleep(120)"],
                             stdout=subprocess.PIPE)
    write_pids(ARGS[0], grandchild=int(grand.stdout.readline()))
    time.sleep(120)
'''


def _alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _survivors(processes, timeout):
    deadline = time.monotonic() + timeout
    while True:
        alive = [process for process in processes if _alive(process)]
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(0.05)


def _kill(processes):
    for process in processes:
        try:
            process.kill()
        except psutil.Error:
            pass
    _survivors(processes, 5.0)


def _wait_for_file(path, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} never appeared")
        time.sleep(0.05)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class StandInCase(unittest.TestCase):
    """A stand-in tool script, a private temp folder for the notes, and a way to run it."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(directory.cleanup)
        self.work = pathlib.Path(directory.name)
        self.notes = self.work / "tmp"
        self.notes.mkdir()
        self.script = self.work / "stand in tool.py"
        self.script.write_text(STAND_IN, encoding="utf-8")
        self.counter = str(self.work / "counter.txt")

    def environment(self, **extra):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SSN_TOOL_SUPERVISOR")}
        for name in ("TMP", "TEMP", "TMPDIR"):
            environment[name] = str(self.notes)
        environment["PYTHONUTF8"] = "1"
        environment.update(extra)
        return environment

    def command(self, mode, *arguments, flags=()):
        return [sys.executable, *flags, str(self.script), str(SRC_DIR), mode, *arguments]

    def run_tool(self, mode, *arguments, flags=(), stdin="", env=None, timeout=90):
        return subprocess.run(
            self.command(mode, *arguments, flags=flags),
            input=stdin, capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env or self.environment(), timeout=timeout,
        )

    def runs(self):
        with open(self.counter) as handle:
            return int(handle.read())

    def assertNoNotes(self):
        self.assertEqual([], os.listdir(self.notes), "A note file was left in the temp folder.")

    @staticmethod
    def unsigned(code):
        return code & 0xFFFFFFFF if WINDOWS else code


class ExitCodeTests(StandInCase):
    def test_exit_codes_are_passed_on(self):
        for code in (0, 1, 7):
            with self.subTest(code=code):
                result = self.run_tool("exit", str(code))
                self.assertEqual(result.returncode, code)
                self.assertEqual(result.stdout.count("start"), 1)
                self.assertNotIn("[Hardware]", result.stdout)

    @NO_CRASH_ON_MAC
    def test_an_abnormal_exit_without_a_note_is_passed_on_and_not_restarted(self):
        result = self.run_tool("crash_no_note", self.counter)
        self.assertEqual(self.runs(), 1)
        self.assertNotIn("[Hardware]", result.stdout)
        if WINDOWS:
            self.assertEqual(result.returncode & 0xFFFFFFFF, CRASH_UNSIGNED)
        else:
            self.assertEqual(result.returncode, 128 + signal.SIGABRT)
        self.assertNoNotes()

    def test_output_through_a_pipe_and_a_file_and_stdin_reach_the_child(self):
        result = self.run_tool("echo", stdin="hello input\n")
        self.assertEqual(result.returncode, 0)
        self.assertIn("stdin hello input", result.stdout)
        self.assertIn("to stderr", result.stderr)
        target = self.work / "captured.txt"
        with open(target, "w", encoding="utf-8") as handle:
            completed = subprocess.run(
                self.command("echo"), input="from stdin\n", text=True, stdout=handle,
                stderr=subprocess.DEVNULL, env=self.environment(), timeout=90,
            )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("stdin from stdin", target.read_text(encoding="utf-8"))

    def test_interpreter_flags_and_awkward_arguments_survive(self):
        awkward = ["two words", "café α.txt", "--flag=x y"]
        result = self.run_tool("echo", *awkward, flags=("-O",))
        self.assertEqual(result.returncode, 0)
        self.assertIn("flags 1 ", result.stdout)
        shown = json.loads(result.stdout.split("argv ", 1)[1].splitlines()[0])
        self.assertEqual(shown, awkward)

    def test_the_off_switch_runs_the_tool_in_one_process(self):
        result = self.run_tool("report", env=self.environment(**{OFF: "0"}))
        self.assertEqual(result.returncode, 0)
        report = json.loads(result.stdout.split("report ", 1)[1])
        self.assertIsNone(report["note"])
        self.assertEqual(report["markers"], [OFF])

    def test_the_child_takes_its_markers_out_of_its_environment(self):
        result = self.run_tool("report")
        report = json.loads(result.stdout.split("report ", 1)[1])
        self.assertIsNotNone(report["note"])
        self.assertEqual(report["markers"], [])

    def test_a_tool_the_child_starts_is_supervised_normally(self):
        result = self.run_tool("nested")
        self.assertEqual(result.returncode, 0)
        inner = json.loads(result.stdout.split("inner report ", 1)[1].splitlines()[0])
        outer = result.stdout.split("outer ", 1)[1].split()[0]
        self.assertIsNotNone(inner["note"])
        self.assertNotEqual(inner["note"], outer)
        self.assertEqual(inner["markers"], [])
        self.assertNoNotes()


class RestartTests(StandInCase):
    @NO_CRASH_ON_MAC
    def test_a_note_and_a_crash_restart_the_tool_once_then_it_succeeds(self):
        result = self.run_tool("crash_once", self.counter)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.runs(), 2)
        self.assertEqual(result.stdout.count("[Hardware] Warning:"), 1)
        self.assertIn("Starting it once more", result.stdout)
        recorded = json.loads(result.stdout.split("second run ", 1)[1].splitlines()[0])
        self.assertEqual(recorded["device"], "cuda:0")
        self.assertEqual(recorded["free_bytes"], 11 * GIB)
        self.assertEqual(recorded["total_bytes"], 12 * GIB)
        self.assertNoNotes()

    @NO_CRASH_ON_MAC
    def test_a_second_crash_is_passed_on_with_a_message_and_not_restarted_again(self):
        result = self.run_tool("crash_always", self.counter)
        self.assertEqual(self.runs(), 2)
        if WINDOWS:
            self.assertEqual(result.returncode & 0xFFFFFFFF, CRASH_UNSIGNED)
        else:
            self.assertEqual(result.returncode, 128 + signal.SIGABRT)
        self.assertEqual(result.stdout.count("[Hardware] Warning:"), 1)
        self.assertIn("ended abnormally again", result.stdout)
        self.assertIn("Finished work is kept", result.stdout)
        self.assertNoNotes()

    @NO_CRASH_ON_MAC
    def test_the_final_message_does_not_need_a_console_that_shows_every_character(self):
        environment = self.environment(PYTHONIOENCODING="cp1252")
        environment.pop("PYTHONUTF8")
        result = self.run_tool("crash_always", self.counter, env=environment)
        self.assertIn("ended abnormally again", result.stdout)

    def test_a_normal_failure_with_a_note_is_not_restarted(self):
        result = self.run_tool("note_exit1", self.counter)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.runs(), 1)
        self.assertNotIn("[Hardware]", result.stdout)
        self.assertNoNotes()

    def test_ctrl_c_with_a_note_is_not_restarted(self):
        result = self.run_tool("note_ctrl_c", self.counter)
        self.assertEqual(self.runs(), 1)
        self.assertNotIn("[Hardware]", result.stdout)
        if WINDOWS:
            self.assertEqual(result.returncode & 0xFFFFFFFF, 0xC000013A)
        else:
            self.assertEqual(result.returncode, -signal.SIGTERM)
        self.assertNoNotes()


class AbnormalExitRuleTests(unittest.TestCase):
    def test_windows_codes(self):
        abnormal = Tool_Supervisor.is_abnormal_exit
        for code in (0xC0000409, 0xC0000005, 0xC0000374, 3, -1073740791):
            with self.subTest(code=code):
                self.assertTrue(abnormal(code, "win32"))
        for code in (0, 1, 2, 130, 0xC000013A, -1073741510, 0x40010005):
            with self.subTest(code=code):
                self.assertFalse(abnormal(code, "win32"))

    def test_posix_signals(self):
        abnormal = Tool_Supervisor.is_abnormal_exit
        for name in ("SIGABRT", "SIGSEGV", "SIGILL", "SIGFPE"):
            with self.subTest(signal=name):
                self.assertTrue(abnormal(-int(getattr(signal, name)), "linux"))
        for number in (-2, -15, -9, -1, 0, 1, 3, 134, 139):
            with self.subTest(code=number):
                self.assertFalse(abnormal(number, "linux"))

    def test_other_values_are_not_crashes(self):
        self.assertFalse(Tool_Supervisor.is_abnormal_exit(None, "win32"))
        self.assertFalse(Tool_Supervisor.is_abnormal_exit(True, "win32"))


class ParentEndedAloneTests(StandInCase):
    def start(self, mode):
        pid_file = str(self.work / "pids.json")
        popen = subprocess.Popen(
            self.command(mode, pid_file), env=self.environment(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
        )
        self.addCleanup(popen.wait)
        pids = _wait_for_file(pid_file)
        # Processes are captured now: behind a venv's redirector the PIDs differ from Popen's.
        processes = {name: psutil.Process(pid) for name, pid in pids.items()}
        self.addCleanup(_kill, [psutil.Process(popen.pid), *processes.values()])
        return popen, processes

    def test_the_tool_ends_within_seconds_when_its_parent_is_ended_alone(self):
        _popen, processes = self.start("sleep")
        self.assertNotEqual(processes["parent"].pid, processes["child"].pid)
        processes["parent"].kill()
        self.assertEqual([], [p.pid for p in _survivors([processes["parent"]], 5.0)])
        survivors = _survivors([processes["child"]], 8.0)
        self.assertEqual([], [p.pid for p in survivors], "The tool outlived its supervising parent.")

    def test_what_the_tool_started_ends_with_it(self):
        _popen, processes = self.start("sleep_grand")
        processes["parent"].kill()
        survivors = _survivors([processes["child"], processes["grandchild"]], 8.0)
        self.assertEqual([], [p.pid for p in survivors], "A process of the tool outlived its parent.")

    def test_the_watcher_ignores_a_parent_that_is_alive(self):
        ended = []
        thread = Tool_Supervisor._watch_parent(os.getpid(), interval=0.05, on_gone=lambda: ended.append(1))
        time.sleep(0.4)
        self.assertTrue(thread.is_alive())
        self.assertEqual(ended, [])


class JobTreeStopTests(unittest.IsolatedAsyncioTestCase):
    """The job manager's own tree stop ends a supervised tool, its parent and what it started."""

    async def asyncSetUp(self):
        from mcp_server.pipeline import Pipeline_Jobs
        from tools.tool_helpers.Tool_Pipeline import ToolInvocation, create_settings_snapshot, get_tool_spec

        directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(directory.cleanup)
        self.work = pathlib.Path(directory.name)
        self.script = self.work / "stand_in_tool.py"
        self.script.write_text(STAND_IN, encoding="utf-8")
        self.pid_file = self.work / "pids.json"

        def prepare(tool_id, settings_source, project_root, *, python_executable=None, snapshot_directory=None):
            spec = get_tool_spec(tool_id)
            snapshot = create_settings_snapshot(spec, settings_source, snapshot_directory=snapshot_directory)
            return ToolInvocation(
                tool=spec,
                argv=(python_executable or sys.executable, "-u", str(self.script), str(SRC_DIR), "sleep_grand", str(self.pid_file)),
                cwd=str(self.work), settings_path=snapshot, owns_settings_snapshot=True,
            )

        patcher = mock.patch.object(Pipeline_Jobs, "prepare_headless_invocation", side_effect=prepare)
        patcher.start()
        self.addCleanup(patcher.stop)
        # The job runs the supervised tool, so the tests' own switch is taken away for it.
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in [name for name in os.environ if name.startswith("SSN_TOOL_SUPERVISOR")]:
            del os.environ[name]
        self.manager = Pipeline_Jobs.PipelineJobManager(
            PROJECT_ROOT, termination_grace=1.0, temporary_parent=self.work / "jobs",
        )
        self.addAsyncCleanup(self.manager.close)
        await self.manager.start()

    async def test_cancel_ends_the_parent_the_child_and_the_grandchild(self):
        import asyncio

        job = await self.manager.submit(
            "sanitize_sequences",
            {"DIRECTORIES": {"FASTA_DIR": str(self.work / "outputs")}, "Sanitize_Sequences.py": {}},
        )
        deadline = time.monotonic() + 60
        while not self.pid_file.exists():
            self.assertLess(time.monotonic(), deadline, "The job never recorded its processes.")
            await asyncio.sleep(0.05)
        pids = json.loads(self.pid_file.read_text(encoding="utf-8"))
        self.assertEqual({"child", "parent", "grandchild"}, set(pids))
        processes = [psutil.Process(pid) for pid in dict.fromkeys(pids.values())]
        self.addCleanup(_kill, processes)
        started = time.monotonic()
        cancelled = await self.manager.cancel(job["job_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        survivors = await asyncio.to_thread(_survivors, processes, 1.0 + 5.0 - (time.monotonic() - started))
        self.assertEqual([], [process.pid for process in survivors], "The stop left part of the tree running.")


class NoteAndWaitTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(directory.cleanup)
        self.path = os.path.join(directory.name, "note.json")
        for name, value in (("_note_path", self.path), ("_restart_info", None), ("_wait_done", False), ("_observed", {})):
            patcher = mock.patch.object(Tool_Supervisor, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_a_note_holds_the_device_the_error_and_the_memory(self):
        self.assertTrue(Tool_Supervisor.record_note(
            "cuda", RuntimeError("CUDA error: out of memory"), where="plan test", free_bytes=5, total_bytes=9))
        with open(self.path, encoding="utf-8") as handle:
            note = json.load(handle)
        self.assertEqual(note["device"], "cuda:0")
        self.assertEqual((note["free_bytes"], note["total_bytes"], note["where"]), (5, 9, "plan test"))
        self.assertIn("out of memory", note["error"])

    def test_the_first_note_stays(self):
        Tool_Supervisor.record_note("cuda:0", RuntimeError("one"), free_bytes=5, total_bytes=9)
        Tool_Supervisor.record_note("cuda:0", RuntimeError("two"), free_bytes=1, total_bytes=9)
        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["free_bytes"], 5)

    def test_a_note_without_its_own_reading_uses_the_largest_free_memory_seen(self):
        Tool_Supervisor.observe_gpu_memory("cuda:0", 7, 9)
        Tool_Supervisor.observe_gpu_memory("cuda:0", 3, 9)
        Tool_Supervisor.record_note("cuda:0", RuntimeError("x"))
        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["free_bytes"], 7)

    def test_a_process_that_is_not_a_supervised_child_writes_nothing(self):
        with mock.patch.object(Tool_Supervisor, "_note_path", None):
            self.assertFalse(Tool_Supervisor.record_note("cuda:0", RuntimeError("x")))
        self.assertFalse(os.path.exists(self.path))

    def wait(self, readings, timeout=60.0, device="cuda:0", info=None):
        Tool_Supervisor._restart_info = (
            {"device": "cuda:0", "free_bytes": 11 * GIB, "total_bytes": 12 * GIB} if info is None else info
        )
        clock = {"now": 0.0}
        pending = list(readings)
        lines = []

        def read():
            return pending.pop(0) if len(pending) > 1 else pending[0]

        def sleep(seconds):
            clock["now"] += seconds

        ran = Tool_Supervisor.wait_for_gpu_memory(
            device, read, timeout=timeout, interval=1.0, sleep=sleep, clock=lambda: clock["now"], out=lines.append)
        return ran, lines, clock["now"]

    def test_the_wait_ends_when_the_memory_is_back_and_says_so_in_one_line(self):
        low, high = (2 * GIB, 12 * GIB), (11 * GIB, 12 * GIB)
        ran, lines, waited = self.wait([low, low, high])
        self.assertTrue(ran)
        self.assertEqual(waited, 2.0)
        self.assertEqual(len(lines), 1)
        self.assertIn("11.0 GB free", lines[0])
        self.assertIn("11.0 GB before the failed test", lines[0])

    def test_the_wait_is_bounded_and_the_run_goes_on(self):
        ran, lines, waited = self.wait([(2 * GIB, 12 * GIB)], timeout=5.0)
        self.assertTrue(ran)
        self.assertEqual(waited, 5.0)
        self.assertEqual(len(lines), 1)
        self.assertIn("did not come back within 5 s", lines[0])
        self.assertIn("going on", lines[0])

    def test_the_wait_happens_once_and_only_for_the_recorded_device(self):
        self.assertEqual(len(self.wait([(11 * GIB, 12 * GIB)])[1]), 1)
        again = Tool_Supervisor.wait_for_gpu_memory("cuda:0", lambda: (0, 1), out=lambda _l: self.fail("waited twice"))
        self.assertFalse(again)
        Tool_Supervisor._wait_done = False
        ran, lines, _ = self.wait([(0, 1)], device="cuda:1")
        self.assertEqual((ran, lines), (False, []))

    def test_no_wait_without_a_restart_or_without_a_recorded_reading(self):
        self.assertEqual(self.wait([(0, 1)], info={})[:2], (False, []))
        Tool_Supervisor._restart_info = None
        self.assertFalse(Tool_Supervisor.wait_for_gpu_memory("cuda:0", lambda: (0, 1), out=lambda _l: self.fail("waited")))

    def test_the_wait_survives_a_reader_that_fails(self):
        Tool_Supervisor._restart_info = {"device": "cuda:0", "free_bytes": 11 * GIB, "total_bytes": 12 * GIB}
        calls = []

        def read():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("context lost")
            return (11 * GIB, 12 * GIB)

        clock = {"now": 0.0}
        lines = []
        Tool_Supervisor.wait_for_gpu_memory(
            "cuda:0", read, interval=1.0, sleep=lambda s: clock.update(now=clock["now"] + s),
            clock=lambda: clock["now"], out=lines.append)
        self.assertEqual(len(lines), 1)
        self.assertIn("is back", lines[0])


class HardwareHelperTests(unittest.TestCase):
    def setUp(self):
        import torch
        from utilities import Hardware_Acceleration as Hardware

        self.hardware = Hardware
        self.cuda = Hardware.DeviceCandidate("cuda:0", "GPU", torch.device("cuda:0"), "cuda")
        self.cpu = Hardware.DeviceCandidate("cpu", "CPU", torch.device("cpu"), "cpu")

    def test_out_of_memory_is_told_from_the_text_the_cuda_runtime_gives(self):
        import torch

        self.assertTrue(self.hardware.is_gpu_out_of_memory(torch.AcceleratorError("CUDA error: out of memory")))
        self.assertTrue(self.hardware.is_gpu_out_of_memory(torch.OutOfMemoryError("x")))
        self.assertFalse(self.hardware.is_gpu_out_of_memory(RuntimeError("CUDA error: invalid argument")))
        self.assertFalse(self.hardware.is_gpu_out_of_memory(ValueError("out of memory")))

    def test_the_free_memory_before_a_test_is_read_and_shown_for_a_failure(self):
        with mock.patch("torch.cuda.mem_get_info", return_value=(int(11.2 * GIB), 12 * GIB)):
            memory = self.hardware.begin_gpu_trial(self.cuda)
        self.assertEqual(memory, (int(11.2 * GIB), 12 * GIB))
        with mock.patch.object(self.hardware.Tool_Supervisor, "record_note") as note:
            text = self.hardware.note_gpu_failure(
                self.cuda, RuntimeError("CUDA error: out of memory"), memory, where="here")
        self.assertEqual(text, " (free before the test: 11.2 of 12.0 GB)")
        note.assert_called_once()
        self.assertEqual(note.call_args.kwargs["free_bytes"], int(11.2 * GIB))

    def test_a_failure_that_is_not_out_of_memory_leaves_no_note_but_still_shows_the_memory(self):
        with mock.patch.object(self.hardware.Tool_Supervisor, "record_note") as note:
            text = self.hardware.note_gpu_failure(self.cuda, ValueError("bad"), (GIB, 12 * GIB))
        self.assertIn("free before the test: 1.0 of 12.0 GB", text)
        note.assert_not_called()

    def test_a_cpu_candidate_and_an_unreadable_device_cost_nothing_and_say_nothing(self):
        self.assertIsNone(self.hardware.begin_gpu_trial(self.cpu))
        with mock.patch("torch.cuda.mem_get_info", side_effect=RuntimeError("sticky")):
            self.assertIsNone(self.hardware.begin_gpu_trial(self.cuda))
        self.assertEqual(self.hardware.note_gpu_failure(self.cpu, RuntimeError("out of memory"), None), "")

    def test_the_first_trial_of_a_restarted_tool_waits_for_the_memory(self):
        info = {"device": "cuda:0", "free_bytes": 11 * GIB, "total_bytes": 12 * GIB}
        readings = iter([(2 * GIB, 12 * GIB), (2 * GIB, 12 * GIB), (11 * GIB, 12 * GIB), (11 * GIB, 12 * GIB)])
        with mock.patch.object(Tool_Supervisor, "_restart_info", info), \
                mock.patch.object(Tool_Supervisor, "_wait_done", False), \
                mock.patch("torch.cuda.mem_get_info", side_effect=lambda *_a: next(readings)), \
                mock.patch.object(Tool_Supervisor.time, "sleep") as sleeper, \
                redirect_stdout(io.StringIO()) as output:
            memory = self.hardware.begin_gpu_trial(self.cuda)
        self.assertEqual(memory[0], 11 * GIB)
        self.assertEqual(sleeper.call_count, 2)
        self.assertEqual(output.getvalue().count("[Hardware] GPU memory is back"), 1)


class RestartedPlanningTests(unittest.TestCase):
    """A tool restarted after a GPU out-of-memory crash plans with a fraction of the free memory."""

    INFO = {"device": "cuda:0", "free_bytes": 11 * GIB, "total_bytes": 12 * GIB}

    def restarted(self, info="default"):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(Tool_Supervisor, "_restart_info", self.INFO if info == "default" else info))
        stack.enter_context(mock.patch.object(Tool_Supervisor, "_wait_done", False))
        stack.enter_context(mock.patch.object(Tool_Supervisor, "_reduction_announced", False))
        stack.enter_context(mock.patch.object(Tool_Supervisor.time, "sleep"))
        self.output = io.StringIO()
        stack.enter_context(redirect_stdout(self.output))

    def backend(self, readings):
        import torch
        from Embedding_Alignment_Engine import get_accelerator_backend

        backend = get_accelerator_backend(torch.device("cuda:0"))
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(type(backend), "device_context", lambda _self: contextlib.nullcontext()))
        stack.enter_context(mock.patch("torch.cuda.mem_get_info", side_effect=readings))
        return backend

    def test_the_plan_sees_the_fraction_and_the_line_is_printed_once_for_the_whole_run(self):
        self.assertEqual(Tool_Supervisor.RESTART_MEMORY_FRACTION, 0.5)
        self.restarted()
        backend = self.backend(lambda *_a: (10 * GIB, 12 * GIB))
        self.assertEqual(backend.memory_snapshot().free_bytes, 5 * GIB)
        self.assertEqual(backend.memory_info(), (5 * GIB, 12 * GIB))
        self.assertEqual(backend.memory_snapshot().free_bytes, 5 * GIB)  # a later batch plans from it too
        text = self.output.getvalue()
        self.assertEqual(text.count("After the restart"), 1)
        self.assertIn("use half of the free memory (5.0 of 10.0 GB)", text)

    def test_the_wait_uses_the_real_figure_and_comes_first(self):
        self.restarted()
        readings = iter([(2 * GIB, 12 * GIB), (11 * GIB, 12 * GIB)] + [(11 * GIB, 12 * GIB)] * 4)
        backend = self.backend(lambda *_a: next(readings))
        self.assertEqual(backend.memory_snapshot().free_bytes, 11 * GIB // 2)
        text = self.output.getvalue()
        self.assertLess(text.index("GPU memory is back"), text.index("After the restart"))

    def test_the_diagnostic_and_the_note_keep_the_real_free_memory(self):
        from utilities import Hardware_Acceleration as Hardware

        self.restarted()
        self.backend(lambda *_a: (10 * GIB, 12 * GIB))
        candidate = Hardware.DeviceCandidate("cuda:0", "GPU", __import__("torch").device("cuda:0"), "cuda")
        memory = Hardware.read_free_memory(candidate)
        self.assertEqual(memory, (10 * GIB, 12 * GIB))
        with mock.patch.object(Hardware.Tool_Supervisor, "record_note") as note:
            text = Hardware.note_gpu_failure(candidate, RuntimeError("out of memory"), memory)
        self.assertIn("10.0 of 12.0 GB", text)
        self.assertEqual(note.call_args.kwargs["free_bytes"], 10 * GIB)

    def test_a_plan_made_from_the_reduced_figure_is_smaller(self):
        from Embedding_Alignment_Engine import cuda_memory_plan
        import torch

        ordinary = None
        for restarted in (False, True):
            stack = contextlib.ExitStack()
            with stack:
                if restarted:
                    self.restarted()
                backend = self.backend(lambda *_a: (10 * GIB, 12 * GIB))
                plan = cuda_memory_plan(torch.device("cuda:0"), lanes=1)
            if not restarted:
                ordinary = plan
            else:
                self.assertEqual(plan.free_bytes, 5 * GIB)
                self.assertLess(plan.usable_bytes, ordinary.usable_bytes)
                self.assertLess(plan.tile_cache_bytes, ordinary.tile_cache_bytes)
                self.assertLess(plan.matrix_bytes, ordinary.matrix_bytes)
        self.assertEqual(ordinary.free_bytes, 10 * GIB)

    def test_an_ordinary_run_another_device_or_no_restart_info_change_nothing(self):
        self.restarted(None)
        backend = self.backend(lambda *_a: (10 * GIB, 12 * GIB))
        self.assertEqual(backend.memory_snapshot().free_bytes, 10 * GIB)
        self.assertEqual(self.output.getvalue(), "")
        self.assertEqual(Tool_Supervisor.planning_free_bytes("cuda:0", 7, 9), 7)
        self.restarted({"device": "cuda:1", "free_bytes": 1, "total_bytes": 2})
        self.assertEqual(Tool_Supervisor.planning_free_bytes("cuda:0", 7, 9), 7)
        self.assertEqual(Tool_Supervisor.planning_free_bytes("cuda:1", 8, 9), 4)

    def test_a_tool_run_with_the_switch_off_is_not_a_restarted_child(self):
        # With the switch off nobody sets the restart variable, so the child state stays empty.
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); from utilities import Tool_Supervisor as T; "
             "T.supervise('__main__', ''); print(T.planning_free_bytes('cuda:0', 10, 12))" % str(SRC_DIR)],
            capture_output=True, text=True, env=dict(os.environ, **{OFF: "0"}), timeout=60)
        self.assertEqual(result.stdout.strip(), "10")


class EngineWaitTests(unittest.TestCase):
    def test_the_engines_first_memory_read_waits_in_a_restarted_tool_and_only_once(self):
        import torch
        from Embedding_Alignment_Engine import get_accelerator_backend

        backend = get_accelerator_backend(torch.device("cuda:0"))
        info = {"device": "cuda:0", "free_bytes": 11 * GIB, "total_bytes": 12 * GIB}
        readings = iter([(2 * GIB, 12 * GIB), (11 * GIB, 12 * GIB)] + [(11 * GIB, 12 * GIB)] * 5)
        with mock.patch.object(Tool_Supervisor, "_restart_info", info), \
                mock.patch.object(Tool_Supervisor, "_wait_done", False), \
                mock.patch.object(type(backend), "device_context", lambda _self: contextlib.nullcontext()), \
                mock.patch("torch.cuda.mem_get_info", side_effect=lambda *_a: next(readings)), \
                mock.patch.object(Tool_Supervisor.time, "sleep") as sleeper, \
                redirect_stdout(io.StringIO()) as output:
            first = backend.memory_snapshot()
            second = backend.memory_info()
        # The wait used the real figures; planning then sees the reduced one.
        self.assertEqual(first.free_bytes, 11 * GIB // 2)
        self.assertEqual(second, (11 * GIB // 2, 12 * GIB))
        self.assertEqual(sleeper.call_count, 1)
        self.assertEqual(output.getvalue().count("[Hardware] GPU memory is back"), 1)

    def test_an_ordinary_run_reads_the_memory_without_waiting(self):
        import torch
        from Embedding_Alignment_Engine import get_accelerator_backend

        backend = get_accelerator_backend(torch.device("cuda:0"))
        with mock.patch.object(Tool_Supervisor, "_restart_info", None), \
                mock.patch.object(type(backend), "device_context", lambda _self: contextlib.nullcontext()), \
                mock.patch("torch.cuda.mem_get_info", return_value=(5, 10)), \
                mock.patch.object(Tool_Supervisor.time, "sleep") as sleeper:
            self.assertEqual(backend.memory_info(), (5, 10))
        sleeper.assert_not_called()


class FakeChild:
    """Stands in for subprocess.Popen: ends with the next code of a script, and may leave a note first."""

    def __init__(self, script, env):
        self.env = env
        self.code, note = script.pop(0)
        if note:
            with open(env[Tool_Supervisor.NOTE_VARIABLE], "w", encoding="utf-8") as handle:
                json.dump({"device": "cuda:0", "free_bytes": 11 * GIB, "total_bytes": 12 * GIB}, handle)

    def wait(self, timeout=None):
        return self.code

    def send_signal(self, _signal):
        pass


class RestartDecisionTests(unittest.TestCase):
    """The whole restart rule of _run_parent, on every platform, with no real process."""

    CASES = {
        "win32": {"crash": 0xC0000409, "abort": 3, "ctrl_c": 0xC000013A},
        "linux": {"crash": -signal.SIGABRT, "abort": -signal.SIGSEGV, "ctrl_c": -signal.SIGTERM},
    }

    def setUp(self):
        directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(directory.cleanup)
        self.folder = directory.name

    def run_parent(self, platform, script):
        """Run the parent over a script of (exit code, wrote a note) per start; return (code, starts, output)."""
        starts = []
        real_abnormal, real_describe = Tool_Supervisor.is_abnormal_exit, Tool_Supervisor._describe_exit

        def popen(command, env=None, **_kw):
            child = FakeChild(script, env)
            starts.append(child)
            return child

        output = io.StringIO()
        with mock.patch.object(Tool_Supervisor.subprocess, "Popen", popen), \
                mock.patch.object(Tool_Supervisor.tempfile, "gettempdir", return_value=self.folder), \
                mock.patch.object(Tool_Supervisor, "_forward_stop_signals"), \
                mock.patch.object(Tool_Supervisor, "is_abnormal_exit", lambda code: real_abnormal(code, platform)), \
                mock.patch.object(Tool_Supervisor, "_describe_exit", lambda code: real_describe(code, platform)), \
                redirect_stdout(output):
            code = Tool_Supervisor._run_parent()
        self.assertEqual([], os.listdir(self.folder), "A note was left behind.")
        return code, starts, output.getvalue()

    def test_a_note_and_a_crash_restart_once_then_the_second_run_decides(self):
        for platform, codes in self.CASES.items():
            with self.subTest(platform=platform, case="restart then success"):
                code, starts, text = self.run_parent(platform, [(codes["crash"], True), (0, False)])
                self.assertEqual((code, len(starts)), (0, 2))
                self.assertEqual(text.count("[Hardware] Warning:"), 1)
                self.assertNotIn("❌", text)
                self.assertNotIn(Tool_Supervisor.WAIT_VARIABLE, starts[0].env)
                self.assertEqual(json.loads(starts[1].env[Tool_Supervisor.WAIT_VARIABLE])["free_bytes"], 11 * GIB)
            with self.subTest(platform=platform, case="second crash"):
                code, starts, text = self.run_parent(platform, [(codes["crash"], True), (codes["abort"], True)])
                self.assertEqual((code, len(starts)), (codes["abort"], 2))
                self.assertEqual(text.count("[Hardware] Warning:"), 1)
                self.assertIn("ended abnormally again", text)
            with self.subTest(platform=platform, case="second crash without a note"):
                code, starts, text = self.run_parent(platform, [(codes["crash"], True), (codes["crash"], False)])
                self.assertEqual((code, len(starts)), (codes["crash"], 2))
                self.assertIn("ended abnormally again", text)

    def test_no_restart_without_a_note_for_a_normal_failure_or_for_ctrl_c(self):
        for platform, codes in self.CASES.items():
            for name, script in (
                ("crash without a note", [(codes["crash"], False)]),
                ("exit 1 with a note", [(1, True)]),
                ("exit 0 with a note", [(0, True)]),
                ("ctrl+c with a note", [(codes["ctrl_c"], True)]),
            ):
                with self.subTest(platform=platform, case=name):
                    code, starts, text = self.run_parent(platform, list(script))
                    self.assertEqual((code, len(starts), text), (script[0][0], 1, ""))

    def test_a_child_that_cannot_be_started_leaves_the_tool_to_run_in_this_process(self):
        with mock.patch.object(Tool_Supervisor.subprocess, "Popen", side_effect=OSError("no process")), \
                mock.patch.object(Tool_Supervisor.tempfile, "gettempdir", return_value=self.folder), \
                mock.patch.object(Tool_Supervisor, "_pass_on") as pass_on, \
                mock.patch.dict(os.environ), mock.patch.object(sys, "argv", [__file__]):
            for name in [n for n in os.environ if n.startswith("SSN_TOOL_SUPERVISOR")]:
                del os.environ[name]
            self.assertIsNone(Tool_Supervisor.supervise("__main__", __file__))
        pass_on.assert_not_called()
        self.assertEqual([], os.listdir(self.folder))

    def test_a_restart_that_cannot_be_started_ends_as_the_first_run_did(self):
        first = FakeChild([(0xC0000409, True)], {Tool_Supervisor.NOTE_VARIABLE: os.path.join(self.folder, "n.json")})
        with mock.patch.object(Tool_Supervisor.subprocess, "Popen", side_effect=[first, OSError("no process")]), \
                mock.patch.object(Tool_Supervisor.tempfile, "gettempdir", return_value=self.folder), \
                mock.patch.object(Tool_Supervisor, "_forward_stop_signals"), \
                mock.patch.object(Tool_Supervisor, "is_abnormal_exit", lambda code: True), \
                mock.patch.object(Tool_Supervisor, "_read_note", return_value={"device": "cuda:0"}), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(Tool_Supervisor._run_parent(), 0xC0000409)

    def test_the_stale_note_sweep_keeps_a_note_a_long_run_wrote_days_ago(self):
        old = os.path.join(self.folder, Tool_Supervisor.NOTE_PREFIX + "old.json")
        recent = os.path.join(self.folder, Tool_Supervisor.NOTE_PREFIX + "recent.json")
        for path, age in ((old, 8 * 24 * 3600), (recent, 3 * 24 * 3600)):
            with open(path, "w") as handle:
                handle.write("{}")
            os.utime(path, (time.time() - age, time.time() - age))
        Tool_Supervisor._sweep_stale_notes(self.folder)
        self.assertEqual(os.listdir(self.folder), [os.path.basename(recent)])

    def test_the_signals_that_are_raised_again_are_the_stop_signals_and_never_a_crash_signal(self):
        stops = set(Tool_Supervisor._STOP_SIGNALS)
        self.assertFalse(stops & set(Tool_Supervisor._POSIX_CRASH_SIGNALS))
        self.assertLessEqual({signal.SIGINT, signal.SIGTERM}, stops)

    def test_posix_passes_on_only_a_stop_signal_by_raising_it_again(self):
        for number, raised in ((signal.SIGABRT, False), (signal.SIGSEGV, False),
                               (signal.SIGTERM, True), (signal.SIGINT, True), (9, True)):
            with self.subTest(signal=int(number)), mock.patch.object(Tool_Supervisor.os, "name", "posix"), \
                    mock.patch.object(Tool_Supervisor.os, "kill") as kill, \
                    mock.patch.object(Tool_Supervisor.signal, "signal"), \
                    mock.patch.object(Tool_Supervisor, "_STOP_SIGNALS", (2, 15, 1, 9)), \
                    mock.patch.object(Tool_Supervisor.time, "sleep"):
                with self.assertRaises(SystemExit) as stop:
                    Tool_Supervisor._pass_on(-int(number))
                self.assertEqual(stop.exception.code, 128 + int(number))
                self.assertEqual(kill.called, raised)


TOOLS = (
    "Align_Similarity_Matrix", "Network_Injection", "Generate_Embeddings", "Embedding_MSA", "Embedding_SSEARCH",
)
HEAVY = {"torch", "numpy", "h5py", "pandas", "scipy", "sklearn", "numba", "tqdm", "Cache_Manifest"}


class ToolHookTests(unittest.TestCase):
    def test_every_tool_calls_the_hook_before_its_heavy_imports(self):
        for name in TOOLS:
            with self.subTest(tool=name):
                tree = ast.parse((SRC_DIR / "tools" / f"{name}.py").read_text(encoding="utf-8"))
                hook = None
                heavy_before = []
                bootstrap_before = False
                for index, node in enumerate(tree.body):
                    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                            and ast.unparse(node.value.func) == "Tool_Supervisor.supervise"):
                        self.assertEqual([ast.unparse(a) for a in node.value.args], ["__name__", "__file__"])
                        hook = index
                        break
                    if isinstance(node, ast.Try) and "_bootstrap" in ast.unparse(node):
                        bootstrap_before = True
                    modules = []
                    if isinstance(node, ast.Import):
                        modules = [alias.name.split(".")[0] for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        modules = [node.module.split(".")[0]]
                    heavy_before += [module for module in modules if module in HEAVY]
                self.assertIsNotNone(hook, f"{name} never calls Tool_Supervisor.supervise")
                self.assertEqual(heavy_before, [], f"{name} imports {heavy_before} before the hook")
                self.assertTrue(bootstrap_before, f"{name} calls the hook before _bootstrap")
                imports = [n for n in tree.body[:hook] if isinstance(n, ast.ImportFrom) and n.module == "utilities"]
                self.assertTrue(any(a.name == "Tool_Supervisor" for n in imports for a in n.names))

    def test_the_hook_does_nothing_in_an_imported_module_or_a_spawn_worker(self):
        for name in ("tools.Align_Similarity_Matrix", "__mp_main__", "some_test"):
            with self.subTest(module=name), mock.patch.object(Tool_Supervisor.subprocess, "Popen") as popen, \
                    mock.patch.object(Tool_Supervisor, "_become_child") as become, \
                    mock.patch.dict(os.environ, {Tool_Supervisor.PARENT_VARIABLE: "1"}):
                Tool_Supervisor.supervise(name, __file__)
                popen.assert_not_called()
                become.assert_not_called()

    def test_the_hook_does_nothing_when_the_script_is_run_inside_another_program(self):
        with mock.patch.object(Tool_Supervisor.subprocess, "Popen") as popen, \
                mock.patch.dict(os.environ), mock.patch.object(sys, "argv", ["another_program.py"]):
            os.environ.pop(OFF, None)
            os.environ.pop(Tool_Supervisor.PARENT_VARIABLE, None)
            Tool_Supervisor.supervise("__main__", str(SRC_DIR / "tools" / "Embedding_MSA.py"))
            popen.assert_not_called()

    def test_importing_a_tool_does_not_start_a_supervisor(self):
        # A real import of one tool in a fresh process, with supervision on, must not spawn anything.
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SSN_TOOL_SUPERVISOR")}
        environment["CUDA_VISIBLE_DEVICES"] = ""
        code = (
            "import sys; sys.path.insert(0, %r); import tools.Network_Injection as m, psutil; "
            "print(len(psutil.Process().children(recursive=True)), m.Tool_Supervisor._note_path)"
        ) % str(SRC_DIR)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=environment, timeout=180)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "0 None", result.stderr)


class LineKindTests(unittest.TestCase):
    def test_the_messages_are_classified_sensibly(self):
        from utilities.Line_Kinds import ERROR, WARNING, classify

        restart = (
            "[Hardware] Warning: the tool ended abnormally (exit code 0xC0000409) after a GPU "
            "out-of-memory error. Finished work is kept. Starting it once more."
        )
        final = "❌ Error: the tool ended abnormally again (exit code 0xC0000409) after one restart. Finished work is kept."
        self.assertEqual(classify(restart), WARNING)
        self.assertEqual(classify(final), ERROR)


if __name__ == "__main__":
    unittest.main()
