# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""EMAP-SSN benchmark: time the program's heavy calculations on a fixed public set.

Start it on purpose, from the Tools window (Manual Tools, Benchmark), through
MCP (emapssn_pipeline action start_benchmark), or from a terminal:

    python -u src/resources/benchmark/Run_Benchmark.py [--stages 1,2,3]

It runs up to ten stages on the bundled metallo-beta-lactamase set (see the
README beside this file), each stage in a process of its own, with every
device setting on Auto. It records each stage's time, throughput, CPU, memory
and the choices the program's own Auto benchmarks made, with the machine's
hardware, and writes a report named by the time the run started:
Benchmark_Report_<date>_<time>.txt, in the language the program shows, with
an English .json beside it. The terminal stays English and ends with the
report's path. Reports are never deleted.

Folders: the work folder is this folder, or SSN_BENCHMARK_WORK_DIR when set
(the tests point it at a private folder). Its temp/ folder holds the run's
files and a lock that stops a second run while one is alive; it is wiped when
a run starts and when it ends. Each stage's settings point every folder into
temp/, so the benchmark never reads or changes tools_settings.json or the
user's data. The bundled FASTA files are always read from this folder.

Exit codes: 0 when every stage ran or was skipped for a stated reason, 1 when
a stage failed or the run was interrupted, 2 when the benchmark could not
start (another run holds the lock, the folder is not writable, too little free
disk space).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time

BENCHMARK_DIR = Path(__file__).resolve().parent
SRC_DIR = BENCHMARK_DIR.parents[1]
PROJECT_ROOT = SRC_DIR.parent
SCRIPT_PATH = Path(__file__).resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import psutil  # noqa: E402

from utilities.Localization import JoinedMessage, Message, display_text  # noqa: E402

# The protocol: change it whenever a stage, its settings or the data change,
# so reports made with different protocols are never compared as equals.
PROTOCOL_VERSION = 1

WORK_DIR_VARIABLE = "SSN_BENCHMARK_WORK_DIR"
TEMP_FOLDER = "temp"
LOCK_NAME = "benchmark.lock"
REPORT_PREFIX = "Benchmark_Report_"
REQUIRED_FREE_BYTES = 1 << 30
SAMPLE_SECONDS = 0.25
STOP_GRACE_SECONDS = 10.0

MAIN_SET = "benchmark_sequences.fasta"
INJECTION_SET = "injection_sequences.fasta"
DATASET_NAME = "UniProtKB/Swiss-Prot IPR001279 (metallo-beta-lactamase), release 2026_03"
MSA_PROTEIN = "Hydroxyacylglutathione hydrolase"
MSA_SEQUENCES = 100
SEARCH_QUERY = "B1XD76"  # E. coli's glyoxalase II (GLO2_ECODH), in the main set.
REFERENCE_MODEL = "esm2_t6_8m"
SEED = 42


class CannotStart(Exception):
    """The benchmark cannot start; its Message says why (exit code 2)."""


# =====================================================================
# 1. Folders: the work folder, temp/, the lock and the report names
# =====================================================================

def work_dir():
    """The folder that holds temp/ and the reports."""
    configured = os.environ.get(WORK_DIR_VARIABLE)
    return Path(configured).resolve() if configured else BENCHMARK_DIR


def temp_dir(folder=None):
    return (folder or work_dir()) / TEMP_FOLDER


def check_writable(folder):
    """Raise CannotStart unless the benchmark can create files in folder."""
    probe = folder / f".write-test-{os.getpid()}"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as error:
        raise CannotStart(Message(
            "The benchmark can't write to {folder} ({error}). Run it from a copy of EMAP-SSN you can write to.",
            folder=str(folder), error=str(error),
        )) from error


def check_disk(folder, required=REQUIRED_FREE_BYTES):
    """Raise CannotStart when the drive of folder has less than required bytes free."""
    free = shutil.disk_usage(folder).free
    if free < required:
        raise CannotStart(Message(
            "The benchmark needs {needed} of free disk space on the drive of {folder}, but only {free} is free.",
            needed=format_bytes(required), folder=str(folder), free=format_bytes(free),
        ))
    return free


def _process_start_time(pid):
    try:
        return psutil.Process(int(pid)).create_time()
    except (psutil.Error, OSError, ValueError, TypeError):
        return None


def read_lock(path):
    """The lock's holder as written ({"pid", "started", "host", "time"}), or None."""
    try:
        holder = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return holder if isinstance(holder, dict) else None


def lock_is_alive(holder):
    """Whether the process that wrote holder still runs: same PID, same start time."""
    if not holder:
        return False
    if holder.get("host") not in (None, socket.gethostname()):
        return True  # Another machine's run in a shared folder: it can't be checked from here.
    started = _process_start_time(holder.get("pid"))
    try:
        return started is not None and abs(started - float(holder["started"])) < 2.0
    except (KeyError, TypeError, ValueError):
        return False


def acquire_lock(folder):
    """Create temp/benchmark.lock for this process, taking over a lock whose process is gone.

    Raises CannotStart while another benchmark holds it.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LOCK_NAME
    me = {
        "pid": os.getpid(),
        "started": _process_start_time(os.getpid()),
        "host": socket.gethostname(),
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    for _attempt in range(3):
        try:
            with open(path, "x", encoding="utf-8") as handle:
                json.dump(me, handle)
            return path
        except FileExistsError:
            holder = read_lock(path)
            if holder is None:
                time.sleep(0.5)  # Possibly being written right now.
                holder = read_lock(path)
            if lock_is_alive(holder):
                raise CannotStart(Message(
                    "Another benchmark is running (process {pid}, started {time}). "
                    "Wait for it to finish, then start this one again.",
                    pid=holder.get("pid"), time=holder.get("time", "?"),
                ))
            try:
                path.unlink()  # Its process is gone: a killed run's lock.
            except FileNotFoundError:
                pass
            except OSError as error:
                raise CannotStart(Message(
                    "The benchmark could not replace the stale lock {path} ({error}).",
                    path=str(path), error=str(error),
                )) from error
    raise CannotStart(Message("The benchmark could not take its lock {path}.", path=str(path)))


def release_lock(path):
    try:
        holder = read_lock(path)
        if holder is None or holder.get("pid") == os.getpid():
            Path(path).unlink()
    except OSError:
        pass


def _make_writable_and_retry(function, path, _error):
    os.chmod(path, stat.S_IWRITE)
    function(path)


def wipe_temp(folder, keep=()):
    """Delete everything in folder except the names in keep.

    Raises CannotStart naming what could not be deleted, such as a file
    another program still has open, so a run never starts on old files.
    """
    if not folder.exists():
        return
    problems = []
    for entry in sorted(folder.iterdir()):
        if entry.name in keep:
            continue
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, onexc=_make_writable_and_retry)
            else:
                try:
                    entry.unlink()
                except PermissionError:
                    os.chmod(entry, stat.S_IWRITE)
                    entry.unlink()
        except OSError as error:
            problems.append(f"{entry}: {error}")
    if problems:
        raise CannotStart(Message(
            "The benchmark could not clear its temporary folder {folder}, so it stopped instead of "
            "starting on old files. Close the program that uses them and start again. {problems}",
            folder=str(folder), problems="; ".join(problems),
        ))


def remove_temp(folder, lock):
    """At the end of a run: delete temp/ and the lock, leaving nothing behind if possible."""
    try:
        wipe_temp(folder, keep=(LOCK_NAME,))
    except CannotStart as error:
        print(f"Warning: {error.args[0]}", flush=True)
    release_lock(lock)
    try:
        folder.rmdir()
    except OSError:
        pass


def reserve_report_paths(folder, started):
    """(report .txt, report .json) named by started; a second run in the same second adds _2."""
    stem = REPORT_PREFIX + started.strftime("%Y-%m-%d_%H-%M-%S")
    for number in itertools.count(1):
        name = stem if number == 1 else f"{stem}_{number}"
        text, data = folder / f"{name}.txt", folder / f"{name}.json"
        if data.exists():
            continue
        try:
            with open(text, "x", encoding="utf-8"):
                pass
        except FileExistsError:
            continue
        return text, data
    raise AssertionError("unreachable")


# =====================================================================
# 2. Numbers as the report writes them
# =====================================================================

def format_bytes(count):
    if count is None:
        return "-"
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def format_seconds(seconds):
    if seconds is None:
        return "-"
    if seconds < 0.1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 10:
        return f"{seconds:.2f} s"
    if seconds < 600:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)} min {rest:.0f} s"


def format_number(value):
    """A rate or count with separators and sensible precision, e.g. 4,980 or 12.3."""
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


# =====================================================================
# 3. Running one stage in a process of its own
# =====================================================================

@dataclass
class StageRun:
    """What the parent measured around one stage process."""

    returncode: int | None
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    peak_processes: int
    interrupted: bool = False
    log: Path | None = None


class TreeSampler(threading.Thread):
    """Sample a process tree every SAMPLE_SECONDS: summed RAM, and each process's CPU time.

    Windows keeps no CPU times for processes that have ended, so each
    process's CPU time is taken as last seen, which can miss up to one
    interval of a short-lived process.
    """

    def __init__(self, pid, interval=SAMPLE_SECONDS):
        super().__init__(daemon=True)
        self.interval = interval
        self.cpu = {}
        self.peak_rss = 0
        self.peak_processes = 0
        self.known = {}
        self._halt = threading.Event()  # Not _stop: threading.Thread has its own.
        try:
            self.root = psutil.Process(pid)
        except psutil.Error:
            self.root = None
        self.sample()

    def processes(self):
        if self.root is None:
            return []
        try:
            found = [self.root] + self.root.children(recursive=True)
        except psutil.Error:
            found = [self.root]
        for process in found:
            try:
                self.known.setdefault((process.pid, process.create_time()), process)
            except psutil.Error:
                pass
        return found

    def sample(self):
        rss = count = 0
        for process in self.processes():
            try:
                with process.oneshot():
                    times = process.cpu_times()
                    memory = process.memory_info()
                    key = (process.pid, process.create_time())
            except psutil.Error:
                continue
            self.cpu[key] = times.user + times.system
            rss += memory.rss
            count += 1
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_processes = max(self.peak_processes, count)

    def run(self):
        while not self._halt.wait(self.interval):
            self.sample()

    def stop(self):
        self._halt.set()

    @property
    def cpu_seconds(self):
        return sum(self.cpu.values())

    def kill_all(self):
        """Kill every process of the tree seen so far, the deepest first."""
        self.processes()
        survivors = []
        for process in reversed(list(self.known.values())):
            try:
                if process.is_running():
                    process.kill()
                    survivors.append(process)
            except psutil.Error:
                pass
        psutil.wait_procs(survivors, timeout=5)


def _forward_output(stream, log_path, echo):
    """Copy a child's output to the log file and, when echo is set, to our own stdout."""
    with open(log_path, "ab") as log:
        while True:
            try:
                chunk = stream.read1(65536)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            if echo is not None:
                try:
                    echo.write(chunk)
                    echo.flush()
                except (OSError, ValueError):
                    echo = None


def _echo_stream():
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        return None
    try:
        sys.stdout.flush()
    except (OSError, ValueError):
        return None
    return buffer


def run_stage_process(command, *, cwd, env, log_path, echo=True):
    """Run command, measure its process tree, show and log its output, return a StageRun.

    The child stays in this process's console and process group, so Ctrl+C,
    closing the console or an MCP cancel reach it too. On Ctrl+C here the
    child gets STOP_GRACE_SECONDS to stop by itself, then its whole tree is
    killed.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        command, cwd=str(cwd), env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    sampler = TreeSampler(process.pid)
    sampler.start()
    forwarder = threading.Thread(
        target=_forward_output, args=(process.stdout, log_path, _echo_stream() if echo else None), daemon=True,
    )
    forwarder.start()
    interrupted = False
    try:
        while True:
            # A timeout lets Ctrl+C through: Windows can't interrupt an endless wait.
            try:
                returncode = process.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        interrupted = True
        try:
            returncode = process.wait(timeout=STOP_GRACE_SECONDS)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            returncode = None
        sampler.kill_all()
        if returncode is None:
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                returncode = None
    finally:
        sampler.stop()
        sampler.join(timeout=2)
        forwarder.join(timeout=5)
        if not forwarder.is_alive():
            process.stdout.close()
    sampler.sample()
    return StageRun(
        returncode=returncode,
        wall_seconds=time.perf_counter() - started,
        cpu_seconds=sampler.cpu_seconds,
        peak_rss_bytes=sampler.peak_rss,
        peak_processes=sampler.peak_processes,
        interrupted=interrupted,
        log=log_path,
    )


def log_tail(path, lines=30):
    """The last lines of a stage's log, for the report of a failed stage."""
    try:
        text = Path(path).read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return []
    shown = [line.rstrip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    return shown[-lines:]


# =====================================================================
# 4. The sequence sets and the sizes of the work
# =====================================================================

def read_records(path):
    """(headers, sequences) of a FASTA file, as written."""
    from utilities.Sequence_Utils import read_fasta

    headers, sequences = read_fasta(os.fspath(path))
    return list(headers), list(sequences)


def file_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def accession(header):
    parts = header.split("|")
    return parts[1] if len(parts) > 2 else header.split()[0]


def all_pairs_workload(sequences):
    """The size of an all-against-all run: sequences, residues, pairs and DP cells (sum of Li x Lj)."""
    lengths = [len(sequence) for sequence in sequences]
    total = sum(lengths)
    return {
        "sequences": len(lengths),
        "residues": total,
        "pairs": len(lengths) * (len(lengths) - 1) // 2,
        "cells": (total * total - sum(length * length for length in lengths)) // 2,
    }


def injection_workload(existing, new):
    """New sequences aligned against the existing ones and against each other."""
    old_total = sum(len(sequence) for sequence in existing)
    within = all_pairs_workload(new)
    return {
        "sequences": len(new),
        "residues": within["residues"],
        "pairs": len(new) * len(existing) + within["pairs"],
        "cells": within["residues"] * old_total + within["cells"],
    }


def msa_entries(headers, count=MSA_SEQUENCES):
    """The headers the MSA stage aligns: the first count entries by accession named exactly MSA_PROTEIN."""
    named = [header for header in headers if f" {MSA_PROTEIN} OS=" in header]
    return sorted(named, key=accession)[:count]


def write_fasta(path, headers, sequences):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for header, sequence in zip(headers, sequences):
            handle.write(f">{header}\n{sequence}\n")
    return path


def prepare_inputs(temp, limit=None):
    """The run's input FASTA files and the sizes of their work.

    Without a limit the bundled files are read where they are. A limit, which
    only the tests use, takes the first records of each set into temp/inputs/,
    and the report says the run is not comparable.
    """
    main_path, injection_path = BENCHMARK_DIR / MAIN_SET, BENCHMARK_DIR / INJECTION_SET
    main_headers, main_sequences = read_records(main_path)
    new_headers, new_sequences = read_records(injection_path)
    if limit:
        keep_new = max(2, limit // 10)
        main_headers, main_sequences = main_headers[:limit], main_sequences[:limit]
        new_headers, new_sequences = new_headers[:keep_new], new_sequences[:keep_new]
        main_path = write_fasta(temp / "inputs" / MAIN_SET, main_headers, main_sequences)
        injection_path = write_fasta(temp / "inputs" / INJECTION_SET, new_headers, new_sequences)
    msa = msa_entries(main_headers)
    by_header = dict(zip(main_headers, main_sequences))
    query = next((header for header in main_headers if accession(header) == SEARCH_QUERY), main_headers[0])
    return {
        "main_fasta": str(main_path),
        "injection_fasta": str(injection_path),
        "msa_headers": msa,
        "search_query": query,
        "workload": {
            "main": all_pairs_workload(main_sequences),
            "injection": injection_workload(main_sequences, new_sequences),
            "msa": all_pairs_workload([by_header[header] for header in msa]),
        },
        "dataset": {
            "name": DATASET_NAME,
            "main_sha256": file_sha256(BENCHMARK_DIR / MAIN_SET),
            "injection_sha256": file_sha256(BENCHMARK_DIR / INJECTION_SET),
            "limit": limit,
        },
    }


# =====================================================================
# 5. The machine: hardware, software and the conditions of the run
# =====================================================================

PACKAGES = (
    "torch", "numba", "llvmlite", "numpy", "scipy", "h5py", "transformers", "esm", "PySide6",
    "umap-learn", "pynndescent", "scikit-learn", "networkx", "markov-clustering", "graspologic-native", "psutil",
)


def _git(*arguments):
    """git's output in the project, or None when git or the repository is missing."""
    try:
        completed = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(PROJECT_ROOT), *arguments],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def program_version():
    """The release the code says it is, and the git commit with a dirty flag when .git exists."""
    import ast

    version = None
    try:
        tree = ast.parse((SRC_DIR / "desktop" / "Desktop_App.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "APPLICATION_VERSION" for t in node.targets):
                version = ast.literal_eval(node.value)
    except (OSError, SyntaxError, ValueError):
        pass
    found = {"version": version, "commit": None, "describe": None, "dirty": None}
    if (PROJECT_ROOT / ".git").exists():
        found["commit"] = _git("rev-parse", "HEAD")
        found["describe"] = _git("describe", "--tags", "--always")
        status = _git("status", "--porcelain", "--untracked-files=no")
        found["dirty"] = None if status is None else bool(status)
    return found


def package_versions():
    from importlib import metadata

    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def cpu_name(hardware):
    """The CPU's model name: Detect_GPU's on Windows, /proc/cpuinfo on Linux, sysctl on macOS."""
    names = [name for name in hardware.get("processors") or [] if name]
    if names:
        return names[0]
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith(("model name", "hardware", "processor\t: ")) and ":" in line:
                    value = line.split(":", 1)[1].strip()
                    if value and not value.isdigit():
                        return value
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            completed = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                       capture_output=True, text=True, timeout=5, check=False)
            if completed.returncode == 0 and completed.stdout.strip():
                return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    import platform

    return platform.processor() or None


def numba_threads():
    """The thread count the Numba kernels use: all usable CPUs but two (utilities/Numba_Threads.py)."""
    try:
        from utilities import Numba_Threads

        return {"threads": Numba_Threads.default_thread_count(), "usable_cpus": Numba_Threads.usable_cpu_count()}
    except Exception:
        return {"threads": None, "usable_cpus": None}


def other_emapssn_processes():
    """Other EMAP-SSN programs running now, which would take a share of the machine."""
    mine = {os.getpid()}
    try:
        mine.update(process.pid for process in psutil.Process().parents())
    except psutil.Error:
        pass
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.info["pid"] in mine or "python" not in (process.info.get("name") or "").lower():
                continue
            # The script a Python process runs is its first argument that names a .py file;
            # code passed with -c is not a script, whatever it mentions.
            script = next((part for part in (process.info.get("cmdline") or [])[1:]
                           if part.lower().endswith(".py")), None)
        except (psutil.Error, TypeError):
            continue
        if script is None:
            continue
        path = Path(script)
        if (path.name.startswith("EMAPSSN_") or path.name in ("Layout_Cache_Generator.py", "Run_Benchmark.py")
                or path.parent.name == "tools"):
            found.append(path.name)
    return sorted(set(found))


def warm_caches():
    """Whether compiled kernels from earlier runs exist, which makes a run faster."""
    numba_files = sum(1 for _ in SRC_DIR.glob("**/__pycache__/*.nbi"))
    gpu_folder = os.environ.get("SSN_LAYOUT_GPU_KERNEL_CACHE")
    if not gpu_folder:
        base = os.environ.get("LOCALAPPDATA") if sys.platform == "win32" else os.environ.get("XDG_CACHE_HOME")
        base = base or str(Path.home() / (".cache" if sys.platform != "win32" else "AppData/Local"))
        gpu_folder = str(Path(base) / "EMAP-SSN" / "gpu_kernels")
    gpu_files = sum(1 for path in Path(gpu_folder).glob("**/*") if path.is_file()) if Path(gpu_folder).exists() else 0
    return {"numba_cache_files": numba_files, "gpu_kernel_cache_files": gpu_files}


def describe_machine():
    """Hardware, software and the conditions at the start, as plain data for the report."""
    import platform

    try:
        import Detect_GPU

        hardware = Detect_GPU.detect_hardware()
    except Exception as error:
        hardware = {"error": f"{type(error).__name__}: {error}"}
    try:
        from mcp_server.pipeline.Compute_Capabilities import discover_compute_capabilities

        capabilities = discover_compute_capabilities(str(PROJECT_ROOT))
    except Exception as error:
        capabilities = {"status": "unavailable", "errors": [{"reason": f"{type(error).__name__}: {error}"}]}
    memory = psutil.virtual_memory()
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    devices = []
    for device in capabilities.get("devices") or []:
        memory_info = device.get("memory") or {}
        capability = device.get("capabilities") or {}
        devices.append({
            "spec": device.get("device_selection"),
            "name": device.get("name"),
            "backend": device.get("backend"),
            "memory_bytes": memory_info.get("total_bytes") if memory_info.get("kind") != "system" else None,
            "tf32": (capability.get("tf32") or {}).get("status"),
            "bf16": (capability.get("bf16") or {}).get("status"),
        })
    nvidia = hardware.get("nvidia_devices") or []
    return {
        "hardware": {
            "cpu": cpu_name(hardware),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "ram_bytes": memory.total,
            "devices": devices,
            "gpus": [
                {"name": gpu.get("name"), "compute_capability": gpu.get("compute_capability"),
                 "driver": gpu.get("driver_version")}
                for gpu in nvidia
            ] or [{"name": name} for name in hardware.get("controllers") or []],
            "os": platform.platform(),
            "os_details": hardware.get("os"),
            "backend": hardware.get("backend"),
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": capabilities.get("pytorch_version"),
            "cuda": capabilities.get("cuda_version"),
            "rocm": capabilities.get("rocm_version"),
            "packages": package_versions(),
            "numba": numba_threads(),
            "backend_profile": _installed_backend_profile(),
        },
        "conditions": {
            "cpu_load_percent": psutil.cpu_percent(interval=1.0),
            "available_ram_bytes": memory.available,
            "on_battery": None if battery is None else not battery.power_plugged,
            "other_emapssn_processes": other_emapssn_processes(),
            "caches": warm_caches(),
        },
    }


def _installed_backend_profile():
    """The PyTorch backend the installer chose, from <sys.prefix>/ssn_backend.json."""
    try:
        state = json.loads((Path(sys.prefix) / "ssn_backend.json").read_text(encoding="utf-8"))
        active = state.get("active_backend") or {}
        return active.get("profile") or active.get("backend")
    except (OSError, ValueError, AttributeError):
        return None


# =====================================================================
# 6. The report: plain data for the .json, words for the .txt
# =====================================================================

STATUS_TEXT = {
    "completed": Message("Completed"),
    "failed": Message("Failed"),
    "skipped": Message("Skipped"),
    "interrupted": Message("Interrupted"),
    "not_run": Message("Not run"),
    "not_selected": Message("Not selected"),
}


def display_width(text):
    """Columns text takes in a terminal or a monospaced file: wide and full-width characters count 2."""
    import unicodedata

    width = 0
    for character in text:
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in ("W", "F") else 1
    return width


def pad(text, width):
    return text + " " * max(0, width - display_width(text))


def table(rows, indent="", gap=2):
    """Lines of rows (lists of cells), each column padded to its widest cell by display width."""
    if not rows:
        return []
    columns = max(len(row) for row in rows)
    widths = [max(display_width(row[index]) if index < len(row) else 0 for row in rows) for index in range(columns)]
    lines = []
    for row in rows:
        cells = [pad(cell, widths[index]) for index, cell in enumerate(row[:-1])] + list(row[-1:])
        lines.append((indent + (" " * gap).join(cells)).rstrip())
    return lines


def heading(text, underline="-"):
    return [text, underline * max(3, display_width(text))]


THROUGHPUT_TEXT = {
    "sequences/s": lambda value: Message("{rate} sequences/s", rate=format_number(value)),
    "residues/s": lambda value: Message("{rate} residues/s", rate=format_number(value)),
    "pairs/s": lambda value: Message("{rate} pairs/s", rate=format_number(value)),
    "billion DP cells/s": lambda value: Message("{rate} billion DP cells/s", rate=format_number(value)),
    "steps/s": lambda value: Message("{rate} layout steps/s", rate=format_number(value)),
    "targets/s": lambda value: Message("{rate} targets/s", rate=format_number(value)),
    "queries/s": lambda value: Message("{rate} queries/s", rate=format_number(value)),
    "edges/s": lambda value: Message("{rate} edges/s", rate=format_number(value)),
    "sequences/s after the Auto trials": lambda value: Message(
        "{rate} sequences/s after the Auto trials", rate=format_number(value)),
    "pairs/s after the Auto trials": lambda value: Message(
        "{rate} pairs/s after the Auto trials", rate=format_number(value)),
}


def throughput_text(item):
    shown = THROUGHPUT_TEXT.get(item.get("unit"))
    return shown(item["value"]) if shown and item.get("value") is not None else "-"


# =====================================================================
# 7. The stages
# =====================================================================

@dataclass(frozen=True)
class Stage:
    """One timed part of the benchmark.

    steps name the functions that run in the stage's own processes, one
    process per step, in order; ready is called in the benchmark's process
    and returns None, or a Message saying why the stage can't run here.
    auto names the kinds of Auto decision that tell the stage's device.
    """

    number: int
    key: str
    title: Message
    needs: tuple = ()
    steps: tuple = ()
    ready: object = None
    auto: tuple = ()


@dataclass
class StageResult:
    """One stage's outcome, as the report tells it."""

    stage: Stage
    status: str = "not_run"
    reason: object = None
    run: StageRun | None = None
    child: dict = field(default_factory=dict)
    decisions: list = field(default_factory=list)
    log_tail: list = field(default_factory=list)


def _missing_packages(*modules):
    """A Message naming the modules Python can't find, or None."""
    import importlib.util

    missing = []
    for name in modules:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    if not missing:
        return None
    return Message(
        "This stage needs %n Python package(s) that are not installed ({packages}).",
        n=len(missing), packages=", ".join(missing),
    )


def umap_problem(state):
    return _missing_packages("umap")


def clustering_problem(state):
    return _missing_packages("graspologic_native", "markov_clustering")


def msa_problem(state):
    count = len(state["inputs"]["msa_headers"])
    if count < 3:
        return Message(
            "The MSA needs at least 3 {protein} sequences, but this run has {count}.",
            count=count, protein=MSA_PROTEIN,
        )
    return None


def blast_executables():
    """blastp and makeblastdb where Align_Substitution_Matrix.py finds them without a BLASTP_DIR.

    The benchmark never reads tools_settings.json, so a BLASTP_DIR saved in
    the Tools window doesn't apply: BLAST+ must be on the PATH or in its
    default Windows folder.
    """
    if shutil.which("blastp") or shutil.which("blastp.exe"):
        return ["blastp", "makeblastdb"]
    root = r"C:\Program Files\NCBI"
    if os.name == "nt" and os.path.isdir(root):
        folders = sorted(
            (os.path.join(root, name, "bin") for name in os.listdir(root)
             if os.path.exists(os.path.join(root, name, "bin", "blastp.exe"))),
            reverse=True,
        )
        if folders:
            return [os.path.join(folders[0], name) for name in ("blastp.exe", "makeblastdb.exe")]
    return ["blastp", "makeblastdb"]


def blast_problem(state):
    for executable in blast_executables():
        try:
            subprocess.run(
                [executable, "-version"], check=True, capture_output=True, timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as error:
            return Message(
                "NCBI BLAST+ was not found on the PATH or in {folder} ({error}).",
                folder=r"C:\Program Files\NCBI", error=f"{executable}: {error}",
            )
    return None


def model_problem(state):
    error = state["model"].get("error")
    if error:
        return Message("The reference model ESM-2 8M could not be downloaded ({error}).", error=error)
    return None


# The ten stages, in the order they run. needs names the stages whose files
# a stage uses; a stage whose needs didn't complete is skipped.
STAGES = (
    Stage(1, "sanitize", Message("Sanitize sequences"), (), ("sanitize",)),
    Stage(2, "embeddings", Message("Embeddings (ESM-2 8M)"), ("sanitize",), ("embeddings",),
          ready=model_problem, auto=("embedding_device",)),
    Stage(3, "alignment", Message("All-against-all embedding alignment"), ("embeddings",), ("alignment",),
          auto=("alignment_plan",)),
    Stage(4, "ssn_layout", Message("SSN layout"), ("alignment",), ("ssn_layout",), auto=("layout_device",)),
    Stage(5, "umap_layout", Message("UMAP layout"), ("alignment",), ("umap_layout",), ready=umap_problem),
    Stage(6, "clustering", Message("Clustering (Leiden, MCL and Jaccard)"), ("alignment",), ("clustering",),
          ready=clustering_problem),
    Stage(7, "search", Message("Embedding database search"), ("embeddings",), ("search",), auto=("search_plan",)),
    Stage(8, "injection", Message("Injection of new sequences"), ("embeddings", "alignment"),
          ("injection_embeddings", "injection_network"), auto=("injection_plan",)),
    Stage(9, "msa", Message("Embedding MSA"), ("embeddings", "alignment"), ("msa",),
          ready=msa_problem, auto=("msa_device",)),
    Stage(10, "blast", Message("BLAST all-against-all alignment"), ("sanitize",), ("blast",), ready=blast_problem),
)
STAGES_BY_KEY = {stage.key: stage for stage in STAGES}
STAGES_BY_NUMBER = {stage.number: stage for stage in STAGES}


# =====================================================================
# 8. Inside a stage's process
# =====================================================================

def _directories(context):
    """Every folder a tool may use, inside temp/."""
    temp = Path(context["temp"])
    return {
        "EMBED_DIR": str(temp / "embeddings"),
        "FASTA_DIR": str(temp / "fasta"),
        "MSA_DIR": str(temp / "msa"),
        "NETWORK_DIR": str(temp / "networks"),
        "REPORT_DIR": str(temp / "reports"),
        "SETTING_EXPORT_DIR": str(temp / "exported_settings"),
    }


def run_tool(context, script_name, tool_settings):
    """Run one src/tools script's main() with explicit settings, as an MCP job does.

    The settings snapshot names every folder inside temp/. The tool module
    is imported, never run as __main__, so the pool workers it spawns can
    find its functions by module name; they inherit the snapshot through
    SSN_TOOL_SETTINGS_SCRIPT and SSN_TOOL_SETTINGS_FILE, which also make the
    import read the snapshot instead of tools_settings.json.
    """
    import importlib

    document = {"DIRECTORIES": _directories(context), script_name: dict(tool_settings)}
    for folder in document["DIRECTORIES"].values():
        Path(folder).mkdir(parents=True, exist_ok=True)
    snapshot = Path(context["temp"]) / "settings" / f"stage_{context['stage']:02d}_{script_name[:-3]}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps(document, indent=4) + "\n", encoding="utf-8")
    os.environ["SSN_TOOL_SETTINGS_SCRIPT"] = script_name
    os.environ["SSN_TOOL_SETTINGS_FILE"] = str(snapshot)
    imported = time.perf_counter()
    module = importlib.import_module(f"tools.{script_name[:-3]}")
    started, started_at = time.perf_counter(), time.time()
    try:
        code = module.main([str(snapshot)])
    except SystemExit as stop:
        code = stop.code
        if isinstance(code, str):
            print(code, flush=True)
            code = 1
    elapsed = time.perf_counter() - started
    return {
        "returncode": int(code or 0),
        "tool_seconds": elapsed,
        "import_seconds": started - imported,
        "started_at": started_at,
        "finished_at": time.time(),
        "settings": {script_name: dict(tool_settings)},
    }


# The fixed protocol: the edge filter the layout and clustering stages use
# (the code's own example), with every other setting at the tools' defaults.
TOP_EDGE_PERCENT = 5.0
ALIGNMENT_SCORE = "global"
NORM_MODE = "alignment_length"
COMBINED_SET = "benchmark_plus_injection.fasta"


def _output(context, key):
    path = context["outputs"].get(key)
    if not path or not Path(path).is_file():
        raise RuntimeError(f"The earlier stage's output {key} is missing.")
    return path


def run_sanitize(context):
    result = run_tool(context, "Sanitize_Sequences.py", {"INPUT_FASTA": context["inputs"]["main_fasta"]})
    stem = Path(context["inputs"]["main_fasta"]).stem
    result["outputs"] = {"sanitized_fasta": str(Path(_directories(context)["FASTA_DIR"]) / f"{stem}_sanitized.fasta")}
    return result


def run_embeddings(context):
    fasta = _output(context, "sanitized_fasta")
    result = run_tool(context, "Generate_Embeddings.py", {"INPUT_FASTA": fasta, "MODEL_NAME": REFERENCE_MODEL})
    name = f"{Path(fasta).stem}_[{REFERENCE_MODEL}]_embeddings.h5"
    result["outputs"] = {"embeddings": str(Path(_directories(context)["EMBED_DIR"]) / name)}
    return result


def run_alignment(context):
    embeddings = _output(context, "embeddings")
    result = run_tool(context, "Align_Similarity_Matrix.py", {"INPUT_HDF5": embeddings})
    name = Path(embeddings).name.replace("_embeddings.h5", "_network.h5")
    result["outputs"] = {"network": str(Path(_directories(context)["NETWORK_DIR"]) / name)}
    return result


def _layout_settings(context, umap):
    from desktop.Viewer_State import encode_document

    flat = {
        "NODE_FASTA_FILE": _output(context, "sanitized_fasta"), "INPUT_HDF5": _output(context, "network"),
        "ALIGNMENT_SCORE": ALIGNMENT_SCORE, "NORM_MODE": NORM_MODE,
        "SIMILARITY_THRESHOLD": None, "TOP_EDGE_PERCENT": None if umap else TOP_EDGE_PERCENT,
        "UMAP_MODE": umap, "UMAP_NEIGHBORS": 15, "UMAP_MIN_DIST": 0.1,
        "LAYOUT_DIMENSIONS": 2, "LAYOUT_SEED": SEED,
        "LAYOUT_DEVICE_SELECTION": "auto", "DT": 0.005, "AUTO_DT": False, "MAX_STEPS": 10000,
        "RMSD_THRESHOLD": 0.005, "PERCENTAGE_DROP_THRESHOLD": 0.1, "RMSD_WINDOW": 50,
        "ENABLE_PROGRESSIVE_SIMULATION": False,
        "SPRING_K": 5.0, "COULOMB_K": 10.0, "COULOMB_CUTOFF": 30.0, "DAMPING": 0.9,
        "MAX_FORCE_LIMIT": 20.0, "MAX_TOTAL_REPULSION_FORCE": 0.0,
        "PACKING_GEOMETRY": "Square", "PACKING_GRID_SIZE": 10.0, "BOX_SCALE": 2.0, "PACKING_PADDING": 10.0,
        "SAVED_LAYOUT_DIR": str(Path(context["temp"]) / "layouts"), "TARGET_CACHE_PATH": None,
        "CACHE_FILENAME": "version_00.h5", "CACHE_NAME_MODE": "auto",
    }
    return flat, encode_document("layout", flat)


def _run_layout(context, umap):
    """generate_layout_cache on stage 3's network, as an MCP layout job does, with the device on Auto."""
    imported = time.perf_counter()
    from Layout_Cache_Generator import LayoutGenerationSettings, generate_layout_cache

    flat, document = _layout_settings(context, umap)
    settings = LayoutGenerationSettings.from_document(document, project_root=str(PROJECT_ROOT))
    started, started_at = time.perf_counter(), time.time()
    result = generate_layout_cache(settings)
    elapsed = time.perf_counter() - started
    return {
        "returncode": 0,
        "tool_seconds": elapsed,
        "import_seconds": started - imported,
        "started_at": started_at,
        "finished_at": time.time(),
        "settings": {"layout": flat},
        "details": {
            "nodes": len(result.full_headers),
            "edges": int(len(result.edges)),
            "effective_similarity_threshold": result.effective_similarity_threshold,
        },
    }


def run_ssn_layout(context):
    return _run_layout(context, umap=False)


def run_umap_layout(context):
    return _run_layout(context, umap=True)


def run_clustering(context):
    """The cluster command's three cores, on the network the Viewer would build at the fixed filter.

    The command itself needs a Viewer, so this runs what it runs: Leiden
    (resolution 1.0, seed 42), MCL (inflation 2.0) and the Jaccard filter
    (0.2), each keeping clusters of at least 10, on stage 3's network with
    the same score, normalization and TOP_EDGE_PERCENT as the SSN layout.
    """
    import contextlib
    import io
    from types import SimpleNamespace
    import warnings

    imported = time.perf_counter()
    import h5py
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse import SparseEfficiencyWarning

    from desktop.Viewer_State import prepare_network
    from utilities import Network_Kernels
    import graspologic_native  # noqa: F401  (imported here so its load is not timed)
    from markov_clustering import mcl  # noqa: F401

    started = time.perf_counter()
    settings = SimpleNamespace(
        ALIGNMENT_SCORE=ALIGNMENT_SCORE, NORM_MODE=NORM_MODE, SIMILARITY_THRESHOLD=None,
        TOP_EDGE_PERCENT=TOP_EDGE_PERCENT, UMAP_MODE=False, UMAP_NEIGHBORS=15, NODE_FASTA_FILE="",
    )
    with h5py.File(_output(context, "network"), "r") as data, contextlib.redirect_stdout(io.StringIO()):
        selected = [header.decode() if isinstance(header, bytes) else str(header) for header in data["headers"][:]]
        headers, edges, scores = prepare_network(data, settings=settings, selected_fasta_headers=selected)
    nodes = len(headers)
    edges = np.asarray(edges, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    prepared = time.perf_counter()

    def clusters(labels):
        return int(len(set(int(label) for label in labels) - {-1}))

    timings, found = {}, {}
    begin = time.perf_counter()
    labels = Network_Kernels.leiden_partition(nodes, edges, scores, 1.0, 10, seed=SEED)
    timings["leiden"], found["leiden"] = time.perf_counter() - begin, clusters(labels)

    warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
    begin = time.perf_counter()
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    matrix = sp.csr_matrix((np.concatenate([scores, scores]), (rows, columns)), shape=(nodes, nodes))
    groups = Network_Kernels.markov_clusters(matrix, 2.0)
    timings["mcl"], found["mcl"] = time.perf_counter() - begin, sum(1 for group in groups if len(group) >= 10)

    # Compile (or load) the Numba Jaccard kernel first, as the Viewer has by its second use.
    Network_Kernels.jaccard_partition(3, np.array([[0, 1], [1, 2]], dtype=np.int32), 0.2, 1)
    begin = time.perf_counter()
    labels = Network_Kernels.jaccard_partition(nodes, edges, 0.2, 10)
    timings["jaccard"], found["jaccard"] = time.perf_counter() - begin, clusters(labels)

    print(f"Clustering {nodes} nodes and {len(edges)} edges (top {TOP_EDGE_PERCENT:g}% of pairs):", flush=True)
    for name in ("leiden", "mcl", "jaccard"):
        print(f"  {name}: {found[name]} clusters of 10 or more in {timings[name]:.3f} s", flush=True)
    return {
        "returncode": 0,
        "tool_seconds": sum(timings.values()),
        "import_seconds": started - imported,
        "settings": {"clustering": {
            "ALIGNMENT_SCORE": ALIGNMENT_SCORE, "NORM_MODE": NORM_MODE, "TOP_EDGE_PERCENT": TOP_EDGE_PERCENT,
            "leiden": "resolution 1.0, minimum size 10, seed 42", "mcl": "inflation 2.0, minimum size 10",
            "jaccard": "threshold 0.2, minimum size 10",
        }},
        "details": {
            "nodes": nodes, "edges": int(len(edges)), "prepare_seconds": prepared - started,
            "seconds": timings, "clusters": found,
        },
    }


def run_injection_embeddings(context):
    combined = Path(context["temp"]) / "inputs" / COMBINED_SET
    if not combined.is_file():
        main = Path(context["inputs"]["main_fasta"]).read_bytes()
        new = Path(context["inputs"]["injection_fasta"]).read_bytes()
        combined.parent.mkdir(parents=True, exist_ok=True)
        combined.write_bytes(main + (b"" if main.endswith(b"\n") else b"\n") + new)
    result = run_tool(context, "Embedding_Injection.py", {
        "INPUT_EMBED": _output(context, "embeddings"), "INPUT_FASTA": str(combined),
    })
    name = f"{combined.stem}_[{REFERENCE_MODEL}]_embeddings.h5"
    result["outputs"] = {"injected_embeddings": str(Path(_directories(context)["EMBED_DIR"]) / name)}
    return result


def run_injection_network(context):
    embeddings = _output(context, "injected_embeddings")
    result = run_tool(context, "Network_Injection.py", {
        "OLD_NETWORK": _output(context, "network"), "NEW_EMBEDDINGS": embeddings,
    })
    name = Path(embeddings).name.replace("_embeddings.h5", "_network.h5")
    result["outputs"] = {"injected_network": str(Path(_directories(context)["NETWORK_DIR"]) / name)}
    return result


SEARCH_OUTPUT = "benchmark_search"
MSA_SET = "benchmark_msa_subset.fasta"


def run_search(context):
    """One query, E. coli's glyoxalase II, against all of stage 2's embeddings."""
    result = run_tool(context, "Embedding_SSEARCH.py", {
        "INPUT_EMBED": _output(context, "embeddings"),
        "QUERY_HEADER": context["inputs"]["search_query"],
        "OUTPUT_NAME": SEARCH_OUTPUT,
    })
    report = Path(_directories(context)["REPORT_DIR"]) / f"Report_{SEARCH_OUTPUT}.txt"
    result["outputs"] = {"search_report": str(report)}
    return result


def run_msa(context):
    """The embedding MSA of the glyoxalase II subset, on stage 2's embeddings and stage 3's network."""
    wanted = set(context["inputs"]["msa_headers"])
    headers, sequences = read_records(context["inputs"]["main_fasta"])
    kept = [(header, sequence) for header, sequence in zip(headers, sequences) if header in wanted]
    subset = write_fasta(
        Path(context["temp"]) / "inputs" / MSA_SET,
        [header for header, _ in kept], [sequence for _, sequence in kept],
    )
    result = run_tool(context, "Embedding_MSA.py", {
        "INPUT_FASTA": str(subset),
        "USE_SEQUENCE_FILTER": True,
        "INPUT_EMBED": _output(context, "embeddings"),
        "INPUT_NETWORK": _output(context, "network"),
    })
    name = f"{subset.stem}_[{REFERENCE_MODEL}]_alignment.fasta"
    result["outputs"] = {"msa": str(Path(_directories(context)["MSA_DIR"]) / name)}
    return result


def run_blast(context):
    """BLASTP all against all on the sanitized main set, with the tool's defaults."""
    fasta = _output(context, "sanitized_fasta")
    result = run_tool(context, "Align_Substitution_Matrix.py", {"INPUT_FASTA": fasta})
    name = f"{Path(fasta).stem}_[BLAST]_EValue.h5"
    result["outputs"] = {"blast_network": str(Path(_directories(context)["NETWORK_DIR"]) / name)}
    return result


MODEL_REPOSITORY = "facebook/esm2_t6_8M_UR50D"
MODEL_FILES = ("config.json", "model.safetensors", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json")


def reference_model_cached():
    """Whether ESM-2 8M is in the Hugging Face cache, checked without any download."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    try:
        return all(isinstance(try_to_load_from_cache(MODEL_REPOSITORY, name), str) for name in MODEL_FILES)
    except Exception:
        return False


def run_download_model(context):
    """Load ESM-2 8M once on the CPU, which downloads it into the Hugging Face cache (not timed)."""
    import importlib.util

    plugin_path = SRC_DIR / "resources" / "pLM_models" / "esm2.py"
    spec = importlib.util.spec_from_file_location("benchmark_esm2_plugin", plugin_path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    import torch

    started = time.perf_counter()
    plugin.load_model(REFERENCE_MODEL, torch.device("cpu"))
    return {"returncode": 0, "tool_seconds": time.perf_counter() - started}


# What each step's process runs, by the step name in Stage.steps.
STEP_RUNNERS = {
    "download_model": run_download_model,
    "sanitize": run_sanitize,
    "embeddings": run_embeddings,
    "alignment": run_alignment,
    "ssn_layout": run_ssn_layout,
    "umap_layout": run_umap_layout,
    "clustering": run_clustering,
    "search": run_search,
    "injection_embeddings": run_injection_embeddings,
    "injection_network": run_injection_network,
    "msa": run_msa,
    "blast": run_blast,
}


def gpu_peak_bytes():
    """The most memory PyTorch allocated on one accelerator in this process, or None."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    peaks = []
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            peaks += [torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available() and xpu.is_initialized():
            peaks += [xpu.max_memory_allocated(index) for index in range(xpu.device_count())]
    except Exception:
        pass
    return max(peaks) if peaks else None


def stage_main(key, context_path):
    """A stage's process: run one step, write its result to temp/results/, return its exit code."""
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    runner = STEP_RUNNERS[key]
    result_path = Path(context["temp"]) / "results" / f"{key}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"returncode": 1}
    try:
        result.update(runner(context) or {})
    except Exception as error:
        import traceback

        traceback.print_exc()
        result.update(returncode=1, error=f"{type(error).__name__}: {error}")
    finally:
        result["gpu_peak_bytes"] = gpu_peak_bytes()
        partial = result_path.with_suffix(".partial")
        partial.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        os.replace(partial, result_path)
        sys.stdout.flush()
    return int(result.get("returncode") or 0)


# =====================================================================
# 9. The benchmark's own process: run the stages
# =====================================================================

def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stage_environment(record, number, offline):
    """A stage's environment: the benchmark's own, recording its Auto decisions into record."""
    from utilities import Benchmark_Record

    environment = dict(os.environ)
    for name in ("SSN_TOOL_SETTINGS_SCRIPT", "SSN_TOOL_SETTINGS_FILE",
                 Benchmark_Record.RECORD_VARIABLE, Benchmark_Record.STAGE_VARIABLE):
        environment.pop(name, None)
    if record is not None:
        environment[Benchmark_Record.RECORD_VARIABLE] = str(record)
        environment[Benchmark_Record.STAGE_VARIABLE] = str(number)
    environment.setdefault("MPLBACKEND", "Agg")  # No tool may open a plot window.
    if offline:
        # The model is cached: don't let the timed stages ask the Hub about it.
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def run_child(step, context, temp, *, number=0, record=None, echo=True, offline=False):
    """Run one step in a process of its own: (StageRun, the step's result)."""
    context_path = temp / "context.json"
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    work = temp / "work"
    work.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(SCRIPT_PATH), "--stage", step, "--context", str(context_path)]
    log = temp / "logs" / f"stage_{number:02d}_{step}.log"
    run = run_stage_process(command, cwd=work, env=stage_environment(record, number, offline), log_path=log, echo=echo)
    return run, read_json(temp / "results" / f"{step}.json") or {}


def combine_runs(runs):
    if not runs:
        return None
    return StageRun(
        returncode=runs[-1].returncode,
        wall_seconds=sum(run.wall_seconds for run in runs),
        cpu_seconds=sum(run.cpu_seconds for run in runs),
        peak_rss_bytes=max(run.peak_rss_bytes for run in runs),
        peak_processes=max(run.peak_processes for run in runs),
        interrupted=any(run.interrupted for run in runs),
        log=runs[-1].log,
    )


def combine_children(children):
    """One stage's result from its steps' results: times add up, peaks take the largest."""
    seconds = [child.get("tool_seconds") for child in children.values() if child.get("tool_seconds") is not None]
    imports = [child.get("import_seconds") for child in children.values() if child.get("import_seconds") is not None]
    peaks = [child.get("gpu_peak_bytes") for child in children.values() if child.get("gpu_peak_bytes") is not None]
    combined = {
        "tool_seconds": sum(seconds) if seconds else None,
        "import_seconds": sum(imports) if imports else None,
        "gpu_peak_bytes": max(peaks) if peaks else None,
        "settings": {},
        "details": {},
        "steps": {step: {"tool_seconds": child.get("tool_seconds"), "returncode": child.get("returncode"),
                         "error": child.get("error"), "started_at": child.get("started_at"),
                         "finished_at": child.get("finished_at")} for step, child in children.items()},
    }
    for step, child in children.items():
        combined["settings"].update(child.get("settings") or {})
        if child.get("details"):
            combined["details"].update(child["details"])
    return combined


_LAYOUT_STOP = re.compile(
    r"^\s+- (?:(Converged|Plateau Reached|Not settling) at Step (\d+)|Step limit reached after (\d+) steps)"
)


_MSA_TIME = re.compile(r"^(Tree building|Cluster merging) time: (?:(\d+)h )?(?:(\d+)m )?([\d.]+)s\s*$")


def msa_times(log_path):
    """The MSA's own tree-building and cluster-merging times, in seconds, from its output."""
    try:
        text = Path(log_path).read_bytes().decode("utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return {}
    times = {}
    for line in text.split("\n"):
        match = _MSA_TIME.match(line.strip())
        if match:
            hours, minutes, seconds = match.group(2), match.group(3), match.group(4)
            total = int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds)
            times["tree_seconds" if match.group(1) == "Tree building" else "merge_seconds"] = total
    return times


def layout_stops(log_path):
    """How each layout job stopped, and after how many steps, from the stage's output."""
    try:
        text = Path(log_path).read_bytes().decode("utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return []
    stops = []
    for line in text.split("\n"):
        match = _LAYOUT_STOP.match(line)
        if match:
            steps = int(match.group(2)) + 1 if match.group(2) else int(match.group(3))
            stops.append({"stop": match.group(1) or "Step limit reached", "steps": steps})
    return stops


def run_stage(stage, outcome, context, temp, *, echo=True, offline=False):
    """Run a stage's steps, each in its own process, and fill in outcome."""
    from utilities import Benchmark_Record

    context["stage"] = stage.number
    record = temp / "records" / f"stage_{stage.number:02d}.jsonl"
    record.parent.mkdir(parents=True, exist_ok=True)
    runs, children = [], {}
    outcome.status = "completed"
    for step in stage.steps:
        run, child = run_child(step, context, temp, number=stage.number, record=record, echo=echo, offline=offline)
        runs.append(run)
        children[step] = child
        if run.interrupted:
            outcome.status = "interrupted"
            break
        if run.returncode != 0 or child.get("returncode") not in (0, None) or not child:
            outcome.status = "failed"
            code = run.returncode if run.returncode not in (0, None) else child.get("returncode", "?")
            outcome.reason = Message("The stage stopped with exit code {code}.", code=code)
            outcome.log_tail = log_tail(run.log)
            break
        context["outputs"].update(child.get("outputs") or {})
    outcome.run = combine_runs(runs)
    outcome.child = combine_children(children)
    outcome.decisions = Benchmark_Record.read_records(record)
    if stage.key == "ssn_layout" and outcome.run is not None:
        outcome.child["layout_stops"] = layout_stops(outcome.run.log)
    if stage.key == "msa" and outcome.run is not None:
        outcome.child["details"].update(msa_times(outcome.run.log))


def select_stages(text):
    """The stage keys to run: those named (numbers or keys, comma-separated) and what they need."""
    if not text:
        return [stage.key for stage in STAGES]
    chosen = set()
    for item in str(text).split(","):
        item = item.strip().lower()
        if not item:
            continue
        stage = STAGES_BY_NUMBER.get(int(item)) if item.isdigit() else STAGES_BY_KEY.get(item)
        if stage is None:
            raise CannotStart(Message(
                "There is no stage {stage}. The stages are {stages}.",
                stage=item, stages=", ".join(f"{stage.number} {stage.key}" for stage in STAGES),
            ))
        chosen.add(stage.key)
    added = True
    while added:
        added = False
        for key in sorted(chosen):
            for need in STAGES_BY_KEY[key].needs:
                if need not in chosen:
                    chosen.add(need)
                    added = True
    return [stage.key for stage in STAGES if stage.key in chosen]


# The decisions a stage makes before its real work starts: the time until the
# last of them is the Auto trials' share of the stage.
TRIAL_KINDS = {
    "embeddings": ("embedding_device",),
    "alignment": ("matmul_precision", "alignment_plan"),
    "injection": ("matmul_precision", "injection_plan"),
}
TRIAL_STEPS = {"embeddings": "embeddings", "alignment": "alignment", "injection": "injection_network"}


def trial_split(outcome):
    """(seconds until the last Auto decision, seconds after it) of a stage's tool, or None."""
    key = outcome.stage.key
    times = [decision.get("time") for decision in outcome.decisions
             if decision.get("kind") in TRIAL_KINDS.get(key, ()) and decision.get("time") is not None]
    step = ((outcome.child or {}).get("steps") or {}).get(TRIAL_STEPS.get(key))
    if not times or not step or step.get("started_at") is None or step.get("finished_at") is None:
        return None
    before, after = max(times) - step["started_at"], step["finished_at"] - max(times)
    return (before, after) if before >= 0 and after > 0 else None


def stage_rates(outcome, inputs):
    """A completed stage's throughput, worked out from the known size of its work."""
    seconds = outcome.child.get("tool_seconds")
    if outcome.status != "completed" or not seconds:
        return []
    work = inputs["workload"]
    steps = outcome.child.get("steps") or {}

    def rate(count, elapsed=seconds, unit=""):
        return {"value": count / elapsed if elapsed else None, "unit": unit}

    key = outcome.stage.key
    split = trial_split(outcome)
    if key == "sanitize":
        return [rate(work["main"]["sequences"], unit="sequences/s")]
    if key == "embeddings":
        rates = [rate(work["main"]["sequences"], unit="sequences/s"), rate(work["main"]["residues"], unit="residues/s")]
        if split:
            rates.append(rate(work["main"]["sequences"], split[1], "sequences/s after the Auto trials"))
        return rates
    if key == "alignment":
        rates = [rate(work["main"]["pairs"], unit="pairs/s"), rate(work["main"]["cells"] / 1e9, unit="billion DP cells/s")]
        if split:
            rates.append(rate(work["main"]["pairs"], split[1], "pairs/s after the Auto trials"))
        return rates
    if key == "ssn_layout":
        total = sum(stop["steps"] for stop in outcome.child.get("layout_stops") or [])
        return [rate(total, unit="steps/s")] if total else []
    if key == "search":
        return [rate(work["main"]["sequences"], unit="targets/s")]
    if key == "injection":
        embedded = (steps.get("injection_embeddings") or {}).get("tool_seconds")
        aligned = (steps.get("injection_network") or {}).get("tool_seconds")
        rates = [rate(work["injection"]["sequences"], embedded, "sequences/s"),
                 rate(work["injection"]["pairs"], aligned, "pairs/s")]
        if split:
            rates.append(rate(work["injection"]["pairs"], split[1], "pairs/s after the Auto trials"))
        return rates
    if key == "blast":
        return [rate(work["main"]["sequences"], unit="queries/s")]
    return []


# The plans' variants and the memory profiles, as the report names them. The Tools
# window names the first two "execution modes" too, so both read alike in a language.
PLAN_NAMES = {
    "scalar": Message("scalar"),
    "tiled": Message("tiled"),
    "serial": Message("serial"),
    "pool": Message("pool"),
}
PROFILE_NAMES = {
    "tile-heavy": Message("tile-heavy"),
    "balanced": Message("balanced"),
    "matrix-heavy": Message("matrix-heavy"),
}


def candidate_text(candidate, kind):
    """The device, and for a plan its variant and lanes, of one Auto candidate."""
    device = candidate.get("device") or candidate.get("spec") or "?"
    if kind in ("alignment_plan", "injection_plan", "search_plan") and candidate.get("variant"):
        plan = PLAN_NAMES.get(candidate["variant"], candidate["variant"])
        if candidate.get("backend") == "cpu":
            return Message("{device}, {plan} plan", device=device, plan=plan)
        if candidate.get("profile"):
            return Message("{device}, {plan} plan with the {profile} memory profile, %n lane(s)",
                           device=device, plan=plan, profile=PROFILE_NAMES.get(candidate["profile"],
                                                                                candidate["profile"]),
                           n=int(candidate.get("lanes") or 1))
        return Message("{device}, {plan} plan, %n lane(s)", device=device, plan=plan,
                       n=int(candidate.get("lanes") or 1))
    return device


def stage_device(outcome, machine):
    """What a stage ran on: its Auto decisions' winners, else the CPU or the machine's only device."""
    kinds = outcome.stage.auto
    if not kinds:
        return Message("CPU")
    shown, seen = [], set()
    for decision in outcome.decisions:
        winner = decision.get("winner")
        if decision.get("kind") in kinds and winner is not None:
            text = candidate_text(decision["candidates"][winner], decision["kind"])
            if str(text) not in seen:
                seen.add(str(text))
                shown.append(text)
    if shown:
        return JoinedMessage(shown, separator="; ")
    devices = (machine.get("hardware") or {}).get("devices") or []
    if len(devices) == 1:
        return devices[0].get("name") or "?"
    return Message("not recorded")


# =====================================================================
# 10. The report: the .json's data and the .txt's words
# =====================================================================

REPORT_FORMAT = 1


@dataclass
class BenchmarkRun:
    """Everything one run's report tells."""

    started: datetime
    selected: list
    outcomes: list
    limit: int | None = None
    finished: datetime | None = None
    inputs: dict | None = None
    machine: dict | None = None
    program: dict | None = None
    model: dict = field(default_factory=dict)
    interrupted: bool = False
    error: str | None = None
    language: str | None = None
    temp: Path | None = None
    temp_peak_bytes: int = 0
    text_path: Path | None = None
    json_path: Path | None = None

    @property
    def status(self):
        if self.interrupted:
            return "interrupted"
        if self.error or any(outcome.status == "failed" for outcome in self.outcomes):
            return "failed"
        return "completed"

    @property
    def exit_code(self):
        return 0 if self.status == "completed" else 1

    @property
    def duration(self):
        return (self.finished - self.started).total_seconds() if self.finished else None


def format_count(value):
    return "-" if value is None else f"{int(value):,}"


def stage_metrics(outcome, machine):
    """A stage's times, CPU use and memory peaks, as numbers."""
    run, child = outcome.run, outcome.child or {}
    if run is None:
        return {}
    tool = child.get("tool_seconds")
    logical = ((machine or {}).get("hardware") or {}).get("logical_cpus")
    use = None
    if logical and run.wall_seconds:
        use = 100.0 * run.cpu_seconds / (run.wall_seconds * logical)
    return {
        "wall_seconds": run.wall_seconds,
        "tool_seconds": tool,
        "startup_seconds": run.wall_seconds - tool if tool is not None else None,
        "import_seconds": child.get("import_seconds"),
        "cpu_seconds": run.cpu_seconds,
        "cpu_use_percent": use,
        "peak_rss_bytes": run.peak_rss_bytes,
        "peak_processes": run.peak_processes,
        "gpu_peak_bytes": child.get("gpu_peak_bytes"),
    }


def stage_ran(outcome):
    return outcome.status in ("completed", "failed", "interrupted")


def decision_choice(decision):
    """What an Auto decision chose, as words."""
    kind = decision.get("kind")
    if kind == "matmul_precision":
        return PRECISION_NAMES.get(decision.get("choice"), str(decision.get("choice")))
    if kind == "host_cache":
        return HOST_CACHE_TEXT.get(decision.get("choice"), str(decision.get("choice")))
    winner = decision.get("winner")
    candidates = decision.get("candidates") or []
    if winner is None or not 0 <= winner < len(candidates):
        return Message("no candidate succeeded")
    return candidate_text(candidates[winner], kind)


def report_data(run):
    """The report as plain English data, for the .json and for MCP."""
    inputs = run.inputs or {}
    stages = []
    for outcome in run.outcomes:
        child = outcome.child or {}
        stage = outcome.stage
        stages.append({
            "number": stage.number,
            "key": stage.key,
            "title": str(stage.title),
            "status": outcome.status,
            "reason": None if outcome.reason is None else str(outcome.reason),
            **stage_metrics(outcome, run.machine),
            "throughput": stage_rates(outcome, inputs) if inputs else [],
            "device": str(stage_device(outcome, run.machine or {})) if stage_ran(outcome) else None,
            "auto_choices": [
                {"kind": decision.get("kind"), "choice": str(decision_choice(decision))}
                for decision in outcome.decisions
            ],
            "decisions": outcome.decisions,
            "settings": child.get("settings"),
            "details": child.get("details"),
            "steps": child.get("steps"),
            "layout_stops": child.get("layout_stops"),
            "log_tail": outcome.log_tail,
        })
    return {
        "report_format": REPORT_FORMAT,
        "protocol_version": PROTOCOL_VERSION,
        "status": run.status,
        "exit_code": run.exit_code,
        "error": run.error,
        "started": run.started.isoformat(timespec="seconds"),
        "finished": run.finished.isoformat(timespec="seconds") if run.finished else None,
        "duration_seconds": run.duration,
        "temp_peak_bytes": run.temp_peak_bytes,
        "language": run.language,
        "program": run.program,
        "dataset": inputs.get("dataset"),
        "workload": inputs.get("workload"),
        "reference_model": {"name": REFERENCE_MODEL, "repository": MODEL_REPOSITORY, **run.model},
        "machine": run.machine,
        "stages": stages,
        "notes": [str(note) for note in report_notes(run)],
        "files": {
            "text": str(run.text_path) if run.text_path else None,
            "json": str(run.json_path) if run.json_path else None,
        },
    }


def result_summary(data):
    """The English summary an MCP job returns: report paths, stages, Auto choices, hardware."""
    machine = data.get("machine") or {}
    hardware = machine.get("hardware") or {}
    software = machine.get("software") or {}
    return {
        "status": data["status"],
        "exit_code": data["exit_code"],
        "error": data["error"],
        "report_text": data["files"]["text"],
        "report_json": data["files"]["json"],
        "protocol_version": data["protocol_version"],
        "started": data["started"],
        "duration_seconds": data["duration_seconds"],
        "stages": [
            {
                "number": stage["number"], "key": stage["key"], "title": stage["title"],
                "status": stage["status"], "reason": stage["reason"],
                "seconds": stage.get("wall_seconds"), "throughput": stage["throughput"],
                "device": stage["device"], "auto_choices": stage["auto_choices"],
            }
            for stage in data["stages"]
        ],
        "hardware": {
            "cpu": hardware.get("cpu"),
            "logical_cpus": hardware.get("logical_cpus"),
            "ram_bytes": hardware.get("ram_bytes"),
            "gpus": hardware.get("gpus"),
            "os": hardware.get("os"),
            "python": software.get("python"),
            "pytorch": software.get("pytorch"),
            "cuda": software.get("cuda"),
        },
    }


PRECISION_NAMES = {"tf32": "TF32", "ieee_fp32": "FP32", "bf16": "BF16"}
PRECISION_REASONS = {
    "faster_and_equivalent": Message("TF32 gave the same results and was at least 1.10 times as fast"),
    "too_little_speedup": Message("TF32 was less than 1.10 times as fast"),
    "not_equivalent": Message("TF32 did not give the same results"),
    "no_nvidia_cuda": Message("there is no NVIDIA CUDA device"),
    "too_few_pairs": Message("there were too few pairs to compare the two"),
    "vram_preflight": Message("the trial would not fit in the GPU's memory"),
    "trial_failed": Message("the trial failed"),
}
HOST_CACHE_TEXT = {
    "packed": Message("all embeddings packed in RAM"),
    "tiles": Message("embeddings read in tiles"),
}
DECISION_TITLES = {
    "embedding_device": Message("Device for the embeddings, by predicted job time"),
    "alignment_plan": Message("Alignment plan, by pairs per second in a short trial"),
    "injection_plan": Message("Injection alignment plan, by pairs per second in a short trial"),
    "msa_device": Message("Device for the MSA, by the median time of sample merges"),
    "layout_device": Message("Device for the layout, by estimated time"),
    "search_plan": Message("Search plan, by predicted search time"),
}
# The layout's size classes (Hardware_Acceleration.layout_size_class), as a decision's title names them.
LAYOUT_SIZE_NAMES = {
    "small": Message("small"),
    "medium": Message("medium"),
    "massive": Message("massive"),
}


def value_text(value, unit):
    """A measured value with its unit."""
    if value is None:
        return "-"
    if unit == "s":
        return format_seconds(value)
    shown = THROUGHPUT_TEXT.get(unit)
    if shown:
        return shown(value)
    return f"{format_number(value)} {unit}" if unit else format_number(value)


def decision_blocks(decision):
    """Words and a table for one Auto decision."""
    kind = decision.get("kind")
    if kind == "matmul_precision":
        reason = PRECISION_REASONS.get(decision.get("reason"), str(decision.get("reason")))
        blocks = [("line", Message(
            "Matrix-product precision {choice}, because {reason}.",
            choice=decision_choice(decision), reason=reason,
        ))]
        rates = decision.get("rates") or []
        if rates:
            blocks.append(("table", [[Message("Precision"), Message("Plan"), Message("Measured")]] + [
                [PRECISION_NAMES.get(rate.get("precision"), str(rate.get("precision"))),
                 PLAN_NAMES.get(rate.get("variant"), str(rate.get("variant") or "")),
                 value_text(rate.get("value"), decision.get("unit"))]
                for rate in rates
            ]))
        return blocks
    if kind == "host_cache":
        return [("line", Message(
            "Host cache: {choice} ({size} of embeddings, limit {limit}).",
            choice=decision_choice(decision), size=format_bytes(decision.get("embedding_bytes")),
            limit=format_bytes(decision.get("limit_bytes")),
        ))]
    title = DECISION_TITLES.get(kind, kind or "?")
    if kind == "layout_device" and decision.get("size_class"):
        size = decision["size_class"]
        title = Message("Device for the layout of {size} components, by estimated time",
                        size=LAYOUT_SIZE_NAMES.get(size, size))
    candidates = decision.get("candidates") or []
    ranking = [index for index in decision.get("ranking") or [] if 0 <= index < len(candidates)]
    order = ranking + [index for index in range(len(candidates)) if index not in ranking]
    memory = any(candidate.get("peak_memory_bytes") for candidate in candidates)
    rows = [[Message("Candidate"), Message("Measured")] + ([Message("Peak memory")] if memory else [])
            + [Message("Result")]]
    for place, index in enumerate(order, 1):
        candidate = candidates[index]
        if index == decision.get("winner"):
            result = Message("Chosen")
        elif candidate.get("error"):
            result = Message("Failed ({error})", error=candidate["error"])
        elif index in ranking:
            result = Message("Ranked {place}", place=place)
        else:
            result = "-"
        peak = [format_bytes(candidate.get("peak_memory_bytes")) if candidate.get("peak_memory_bytes") else "-"]
        rows.append([candidate_text(candidate, kind), value_text(candidate.get("value"), decision.get("unit"))]
                    + (peak if memory else []) + [result])
    blocks = [("line", title), ("table", rows)]
    if decision.get("tie_fraction") and tie_reordered(decision, [candidates[index] for index in ranking]):
        blocks.append(("line", Message(
            "Results within {percent} % of the best count as a tie, which goes to the CPU, a scalar plan, "
            "less peak memory and fewer lanes, in that order.",
            percent=f"{decision['tie_fraction'] * 100:g}",
        )))
    return blocks


def tie_reordered(decision, ranked):
    """Whether a ranking puts a candidate above one that measured better, as only a tie
    (Hardware_Acceleration.rank_benchmark_results) does, so the table needs the rule beside it."""
    values = [float(candidate["value"]) for candidate in ranked if candidate.get("value") is not None]
    if decision.get("direction") == "lower":
        values = [-value for value in values]
    return any(later > earlier for place, earlier in enumerate(values) for later in values[place + 1:])


def work_text(outcome, inputs):
    """The size of a stage's work, in words."""
    work = (inputs or {}).get("workload") or {}
    main, injection, msa = work.get("main") or {}, work.get("injection") or {}, work.get("msa") or {}
    key = outcome.stage.key
    if key in ("sanitize", "embeddings"):
        return Message("{sequences} sequences, {residues} residues",
                       sequences=format_count(main.get("sequences")), residues=format_count(main.get("residues")))
    if key == "alignment":
        return Message("{pairs} pairs, {cells} dynamic-programming cells",
                       pairs=format_count(main.get("pairs")), cells=format_count(main.get("cells")))
    if key in ("ssn_layout", "umap_layout", "clustering"):
        return Message("{nodes} nodes", nodes=format_count(main.get("sequences")))
    if key == "search":
        return Message("One query against {sequences} sequences", sequences=format_count(main.get("sequences")))
    if key == "injection":
        return Message("{sequences} new sequences, {pairs} new pairs",
                       sequences=format_count(injection.get("sequences")), pairs=format_count(injection.get("pairs")))
    if key == "msa":
        return Message("{sequences} sequences, {residues} residues",
                       sequences=format_count(msa.get("sequences")), residues=format_count(msa.get("residues")))
    if key == "blast":
        return Message("{sequences} queries against {sequences} sequences", sequences=format_count(main.get("sequences")))
    return "-"


def detail_rows(outcome):
    """Rows of what a stage reported about its own work."""
    child = outcome.child or {}
    details = child.get("details") or {}
    rows = []
    key = outcome.stage.key
    if key in ("ssn_layout", "umap_layout") and details.get("nodes") is not None:
        rows.append([Message("Network"), Message(
            "{nodes} nodes, {edges} edges", nodes=format_count(details["nodes"]), edges=format_count(details.get("edges")),
        )])
    if key == "ssn_layout" and child.get("layout_stops"):
        stops = child["layout_stops"]
        rows.append([Message("Layout jobs"), Message(
            "%n job(s), {steps} steps in all", n=len(stops), steps=format_count(sum(stop["steps"] for stop in stops)),
        )])
    if key == "clustering" and details.get("seconds"):
        rows.append([Message("Network"), Message(
            "{nodes} nodes, {edges} edges", nodes=format_count(details.get("nodes")), edges=format_count(details.get("edges")),
        )])
        for method, name in (("leiden", "Leiden"), ("mcl", "MCL"), ("jaccard", "Jaccard")):
            if method in details["seconds"]:
                rows.append([name, Message(
                    "{clusters} clusters of 10 or more in {time}",
                    clusters=format_count((details.get("clusters") or {}).get(method)),
                    time=format_seconds(details["seconds"][method]),
                )])
    if key == "msa":
        if details.get("tree_seconds") is not None:
            rows.append([Message("Tree building"), format_seconds(details["tree_seconds"])])
        if details.get("merge_seconds") is not None:
            rows.append([Message("Cluster merging"), format_seconds(details["merge_seconds"])])
    if key == "injection":
        steps = child.get("steps") or {}
        for step, label in (("injection_embeddings", Message("Embedding the new sequences")),
                            ("injection_network", Message("Aligning the new sequences"))):
            if (steps.get(step) or {}).get("tool_seconds") is not None:
                rows.append([label, format_seconds(steps[step]["tool_seconds"])])
    return rows


def report_notes(run):
    """Notes that say how far the run's numbers can be compared."""
    notes = []
    if run.interrupted:
        notes.append(Message("The run was interrupted, so the stages after it did not run."))
    if run.error:
        notes.append(Message("The benchmark itself failed ({error}).", error=run.error))
    if run.limit:
        notes.append(Message(
            "This run used only the first %n sequence(s) of the set, a test option, so it can't be compared with full runs.",
            n=run.limit,
        ))
    notes.append(Message("Stage times include the time each tool spends on its own Auto hardware trials."))
    notes.append(Message(
        "ESM-2 8M makes embeddings 320 values wide, so the alignment's matrix products are cheaper here than with larger models."
    ))
    conditions = (run.machine or {}).get("conditions") or {}
    load = conditions.get("cpu_load_percent")
    if load is not None and load >= 20:
        notes.append(Message("The CPU was {load} busy when the run started, which can slow the run down.",
                             load=f"{load:.0f} %"))
    if conditions.get("on_battery"):
        notes.append(Message("The computer ran on battery, which can slow the run down."))
    if conditions.get("other_emapssn_processes"):
        notes.append(Message("Other EMAP-SSN programs were running ({programs}), which can slow the run down.",
                             programs=", ".join(conditions["other_emapssn_processes"])))
    notes.append(Message(
        "A machine's first run also compiles kernels that later runs load from a cache, so it can be slower."
    ))
    notes.append(Message("Compare reports only when their protocol versions and sequence sets match."))
    return notes


def _machine_rows(run):
    machine = run.machine or {}
    hardware, software = machine.get("hardware") or {}, machine.get("software") or {}
    rows = [[Message("CPU"), hardware.get("cpu") or "-"]]
    rows.append([Message("CPU cores"), Message(
        "{logical} logical CPUs, {physical} physical cores",
        logical=format_count(hardware.get("logical_cpus")), physical=format_count(hardware.get("physical_cores")),
    )])
    rows.append([Message("Memory (RAM)"), format_bytes(hardware.get("ram_bytes"))])
    for gpu in hardware.get("gpus") or []:
        parts = [gpu.get("name") or "?"]
        if gpu.get("driver"):
            parts.append(Message("driver {driver}", driver=gpu["driver"]))
        if gpu.get("compute_capability"):
            parts.append(Message("compute capability {capability}", capability=gpu["compute_capability"]))
        rows.append([Message("GPU"), JoinedMessage(parts, separator=", ")])
    for device in hardware.get("devices") or []:
        if device.get("backend") == "cpu":
            continue
        name = f"{device.get('name') or '?'} [{device.get('spec') or '?'}]"
        if device.get("memory_bytes"):
            name = JoinedMessage([name, Message("with {memory}", memory=format_bytes(device["memory_bytes"]))])
        rows.append([Message("Compute device"), name])
    rows.append([Message("Operating system"), hardware.get("os") or "-"])
    rows.append(["Python", software.get("python") or "-"])
    rows.append(["PyTorch", software.get("pytorch") or "-"])
    if software.get("cuda"):
        rows.append(["CUDA", software["cuda"]])
    if software.get("rocm"):
        rows.append(["ROCm", software["rocm"]])
    if software.get("backend_profile"):
        rows.append([Message("Installed backend"), software["backend_profile"]])
    numba = software.get("numba") or {}
    if numba.get("threads"):
        rows.append([Message("Numba threads"), Message(
            "{threads} of {usable} usable CPUs", threads=numba["threads"], usable=numba.get("usable_cpus") or "?",
        )])
    return rows


def _condition_rows(run):
    conditions = (run.machine or {}).get("conditions") or {}
    load = conditions.get("cpu_load_percent")
    battery = conditions.get("on_battery")
    others = conditions.get("other_emapssn_processes") or []
    caches = conditions.get("caches") or {}
    model = run.model or {}
    if model.get("error"):
        model_text = Message("Download failed ({error})", error=model["error"])
    elif model.get("download_seconds") is not None:
        model_text = Message("Downloaded before the timed stages in {time}",
                             time=format_seconds(model["download_seconds"]))
    elif model.get("cached_before"):
        model_text = Message("Already in the Hugging Face cache")
    else:
        model_text = Message("Not needed by the selected stages")
    return [
        [Message("CPU load"), "-" if load is None else f"{load:.0f} %"],
        [Message("Available RAM"), format_bytes(conditions.get("available_ram_bytes"))],
        [Message("Power"), Message("No battery found") if battery is None
         else (Message("Battery") if battery else Message("Mains"))],
        [Message("Other EMAP-SSN programs"), ", ".join(others) if others else Message("None")],
        [Message("Compiled kernel caches"), Message(
            "{numba} Numba files, {gpu} GPU kernel files",
            numba=format_count(caches.get("numba_cache_files")), gpu=format_count(caches.get("gpu_kernel_cache_files")),
        )],
        [Message("Reference model"), model_text],
    ]


def _program_text(program):
    program = program or {}
    version = program.get("version") or "?"
    commit = (program.get("commit") or "")[:10]
    if not commit:
        return Message("EMAP-SSN {version}", version=version)
    if program.get("dirty"):
        return Message("EMAP-SSN {version}, commit {commit}, with local changes", version=version, commit=commit)
    return Message("EMAP-SSN {version}, commit {commit}", version=version, commit=commit)


def setting_text(value, run):
    """A setting as the report shows it, with a path in temp/ or beside this script made short."""
    if isinstance(value, str) and os.path.isabs(value):
        roots = [(run.temp, "temp/")] if run.temp else []
        for root, label in roots + [(BENCHMARK_DIR, "")]:
            try:
                return label + Path(value).relative_to(root).as_posix()
            except ValueError:
                continue
        return value
    return json.dumps(value, ensure_ascii=False)


def report_blocks(run):
    """The report's text as blocks of Messages and plain values, for render()."""
    inputs = run.inputs or {}
    dataset = inputs.get("dataset") or {}
    work = inputs.get("workload") or {}
    counts = {status: sum(1 for outcome in run.outcomes if outcome.status == status)
              for status in ("completed", "failed", "skipped")}
    blocks = [("title", Message("EMAP-SSN benchmark report"))]
    blocks.append(("table", [
        [Message("Result"), STATUS_TEXT[run.status]],
        [Message("Stages"), Message(
            "{completed} completed, {failed} failed, {skipped} skipped",
            completed=counts["completed"], failed=counts["failed"], skipped=counts["skipped"],
        )],
        [Message("Started"), run.started.strftime("%Y-%m-%d %H:%M:%S")],
        [Message("Finished"), run.finished.strftime("%Y-%m-%d %H:%M:%S") if run.finished else "-"],
        [Message("Duration"), format_seconds(run.duration)],
        [Message("Disk space used"), format_bytes(run.temp_peak_bytes)],
        [Message("Program"), _program_text(run.program)],
        [Message("Benchmark protocol"), str(PROTOCOL_VERSION)],
        [Message("Report language"), run.language or "en"],
        [Message("Sequence set"), dataset.get("name") or DATASET_NAME],
        [Message("Sequences"), Message(
            "{main} in the main set, {new} for injection",
            main=format_count((work.get("main") or {}).get("sequences")),
            new=format_count((work.get("injection") or {}).get("sequences")),
        )],
        [Message("Reference model"), f"{REFERENCE_MODEL} ({MODEL_REPOSITORY})"],
    ]))

    blocks.append(("heading", Message("Summary")))
    rows = [["#", Message("Stage"), Message("Status"), Message("Time"), Message("Throughput"), Message("Device")]]
    for outcome in run.outcomes:
        rates = stage_rates(outcome, inputs) if inputs else []
        rows.append([
            str(outcome.stage.number),
            outcome.stage.title,
            STATUS_TEXT[outcome.status],
            format_seconds(outcome.run.wall_seconds) if outcome.run else "-",
            JoinedMessage([throughput_text(rate) for rate in rates], separator="; ") if rates else "-",
            stage_device(outcome, run.machine or {}) if stage_ran(outcome) else "-",
        ])
    blocks.append(("table", rows))
    reasons = [[Message("Stage {number}", number=outcome.stage.number), outcome.reason]
               for outcome in run.outcomes if outcome.reason is not None]
    if reasons:
        blocks.append(("line", ""))
        blocks.append(("table", reasons))

    blocks.append(("heading", Message("Hardware and software")))
    blocks.append(("table", _machine_rows(run)))
    packages = ((run.machine or {}).get("software") or {}).get("packages") or {}
    if packages:
        blocks.append(("subheading", Message("Packages")))
        blocks.append(("table", [[name, version or Message("not installed")] for name, version in packages.items()]))
    blocks.append(("subheading", Message("Conditions at the start")))
    blocks.append(("table", _condition_rows(run)))

    blocks.append(("heading", Message("Auto hardware decisions")))
    decided = [outcome for outcome in run.outcomes if outcome.decisions]
    if not decided:
        blocks.append(("line", Message("No stage recorded an Auto decision.")))
    for outcome in decided:
        blocks.append(("subheading", Message("Stage {number}, {title}", number=outcome.stage.number,
                                             title=outcome.stage.title)))
        for decision in outcome.decisions:
            blocks.extend(decision_blocks(decision))

    blocks.append(("heading", Message("Stage details")))
    for outcome in run.outcomes:
        if outcome.status == "not_selected":
            continue
        blocks.append(("subheading", Message("Stage {number}, {title}", number=outcome.stage.number,
                                             title=outcome.stage.title)))
        metrics = stage_metrics(outcome, run.machine)
        rows = [[Message("Status"), STATUS_TEXT[outcome.status]]]
        if outcome.reason is not None:
            rows.append([Message("Reason"), outcome.reason])
        if metrics:
            rows.append([Message("Total time"), format_seconds(metrics["wall_seconds"])])
            rows.append([Message("Time in the tool"), format_seconds(metrics["tool_seconds"])])
            rows.append([Message("Start-up and imports"), format_seconds(metrics["startup_seconds"])])
            use = metrics["cpu_use_percent"]
            rows.append([Message("CPU time"), Message(
                "{time}, {share} of all logical CPUs on average",
                time=format_seconds(metrics["cpu_seconds"]), share="-" if use is None else f"{use:.0f} %",
            )])
            rows.append([Message("Peak RAM"), Message(
                "{size} in %n process(es)", n=metrics["peak_processes"], size=format_bytes(metrics["peak_rss_bytes"]),
            )])
            peak = metrics["gpu_peak_bytes"]
            rows.append([Message("Peak GPU memory"), format_bytes(peak) if peak else Message("None recorded")])
        split = trial_split(outcome)
        if split:
            rows.append([Message("Auto trials"), Message(
                "{time} until the last Auto decision, {rest} after it",
                time=format_seconds(split[0]), rest=format_seconds(split[1]),
            )])
        rows.append([Message("Work"), work_text(outcome, inputs)])
        rates = stage_rates(outcome, inputs) if inputs else []
        if rates:
            rows.append([Message("Throughput"), JoinedMessage([throughput_text(rate) for rate in rates], separator="; ")])
        rows.extend(detail_rows(outcome))
        blocks.append(("table", rows))
        for script, settings in ((outcome.child or {}).get("settings") or {}).items():
            blocks.append(("line", Message("Settings for {name}", name=script)))
            blocks.append(("table", [[str(name), setting_text(value, run)] for name, value in settings.items()]))
        if outcome.log_tail:
            blocks.append(("line", Message("The last lines of its output")))
            blocks.append(("verbatim", outcome.log_tail))

    blocks.append(("heading", Message("Notes")))
    blocks.append(("bullets", report_notes(run)))
    return blocks


def render(blocks, show):
    """The report's text: show turns each Message into words (str for English, display_text translated)."""
    lines = []
    for kind, content in blocks:
        if kind == "title":
            text = show(content)
            lines += [text, "=" * max(3, display_width(text)), ""]
        elif kind == "heading":
            text = show(content)
            lines += ["", text, "-" * max(3, display_width(text))]
        elif kind == "subheading":
            lines += ["", show(content)]
        elif kind == "line":
            lines.append(show(content))
        elif kind == "table":
            lines += table([[show(cell) for cell in row] for row in content], indent="  ")
        elif kind == "bullets":
            lines += [f"- {show(item)}" for item in content]
        elif kind == "verbatim":
            lines += [f"    {line}" for line in content]
    return "\n".join(lines).rstrip() + "\n"


def english(value):
    return str(value)


def report_language():
    """The language the program shows, read once as the run starts: the one user setting the benchmark reads.

    None means English. A setting that can't be read leaves the report English.
    """
    try:
        from desktop.Desktop_App import startup_language

        return startup_language()
    except Exception as error:
        say(f"Warning: the report will be in English, because the language setting could not be read ({error}).")
        return None


def install_report_language(language):
    """Install language's translations for writing the .txt; return what remove() takes them away, or None.

    The translations need a Qt application. An offscreen QGuiApplication is
    enough, and a bare QCoreApplication is not: it crashes when the Chinese
    font registers. Anything that fails leaves the report English.
    """
    if not language:
        return None
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtGui import QGuiApplication

        from desktop.Desktop_App import install_translations

        app = QGuiApplication.instance() or QGuiApplication([])
        return install_translations(app, language)
    except Exception as error:
        say(f"Warning: the report is in English, because language {language} could not be installed "
            f"({type(error).__name__}: {error}).")
        return None


def write_reports(run, folder):
    """Write the English .json and the .txt in the run's language beside each other; return the data."""
    run.text_path, run.json_path = reserve_report_paths(folder, run.started)
    data = report_data(run)
    partial = run.json_path.with_name(run.json_path.name + ".partial")
    partial.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(partial, run.json_path)
    installed = install_report_language(run.language)
    try:
        text = render(report_blocks(run), display_text)
    finally:
        if installed is not None:
            installed.remove()
    with open(run.text_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return data


# =====================================================================
# 11. The benchmark's run, start to end
# =====================================================================

def folder_bytes(folder):
    """The size of the files in folder and its subfolders."""
    total = 0
    for root, _folders, files in os.walk(folder):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def say(text=""):
    print(text, flush=True)


def write_result(path, summary):
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(partial, target)


def cannot_start(error, result_path):
    reason = error.args[0] if error.args else error
    say(f"The benchmark could not start. {reason}")
    write_result(result_path, {"status": "could_not_start", "exit_code": 2, "reason": str(reason)})
    return 2


def skip(outcome, reason):
    outcome.status, outcome.reason = "skipped", reason
    say(f"\nStage {outcome.stage.number} ({outcome.stage.title}) is skipped: {reason}")


def run_stages(run, temp, echo=True):
    """Check the machine, prepare the inputs and run the selected stages in order."""
    say(f"EMAP-SSN benchmark, protocol {PROTOCOL_VERSION}")
    say(f"Work folder: {temp.parent}")
    say("Stages: " + ", ".join(f"{STAGES_BY_KEY[key].number} {key}" for key in run.selected))
    if run.limit:
        say(f"Test option: only the first {run.limit} sequences; the report can't be compared with full runs.")
    run.language = report_language()
    say(f"Report language: {run.language or 'English'}")
    say("Recording the hardware and software...")
    run.program = program_version()
    run.machine = describe_machine()
    run.inputs = prepare_inputs(temp, run.limit)
    context = {"temp": str(temp), "inputs": run.inputs, "outputs": {}, "stage": 0}

    run.model = {"cached_before": reference_model_cached()}
    if "embeddings" in run.selected and not run.model["cached_before"]:
        say("Downloading the reference model ESM-2 8M into the Hugging Face cache (not timed)...")
        download, child = run_child("download_model", context, temp, number=0, echo=echo)
        if download.interrupted:
            raise KeyboardInterrupt
        run.model["download_seconds"] = download.wall_seconds
        if download.returncode != 0 or child.get("returncode") or not reference_model_cached():
            run.model["error"] = child.get("error") or f"exit code {download.returncode}"
    offline = reference_model_cached()
    state = {"inputs": run.inputs, "model": run.model}

    by_key = {outcome.stage.key: outcome for outcome in run.outcomes}
    chosen = [outcome for outcome in run.outcomes if outcome.status != "not_selected"]
    for index, outcome in enumerate(chosen, 1):
        stage = outcome.stage
        missing = [STAGES_BY_KEY[need] for need in stage.needs if by_key[need].status != "completed"]
        if missing:
            skip(outcome, Message(
                "It needs {stages}, which did not complete.",
                stages=JoinedMessage([Message("stage {number}", number=need.number) for need in missing],
                                     separator=", "),
            ))
            continue
        problem = stage.ready(state) if stage.ready else None
        if problem is not None:
            skip(outcome, problem)
            continue
        say(f"\n=== Stage {stage.number} ({index} of {len(chosen)}): {stage.title} ===")
        run_stage(stage, outcome, context, temp, echo=echo, offline=offline)
        run.temp_peak_bytes = max(run.temp_peak_bytes, folder_bytes(temp))
        took = f" in {format_seconds(outcome.run.wall_seconds)}" if outcome.run else ""
        say(f"=== Stage {stage.number} {outcome.status}{took} ===")
        if outcome.status == "interrupted":
            run.interrupted = True
            break


def _raise_interrupt(_signum, _frame):
    raise KeyboardInterrupt


def run_benchmark(stages=None, result_path=None, limit=None, echo=True):
    """Run the benchmark and write its reports; return the exit code (0, 1 or 2)."""
    started = datetime.now().astimezone()
    folder = work_dir()
    temp = temp_dir(folder)
    try:
        selected = select_stages(stages)
        check_writable(folder)
        lock = acquire_lock(temp)
    except CannotStart as error:
        return cannot_start(error, result_path)
    try:
        wipe_temp(temp, keep=(LOCK_NAME,))
        check_disk(folder)
    except CannotStart as error:
        release_lock(lock)
        return cannot_start(error, result_path)

    run = BenchmarkRun(started=started, selected=selected, limit=limit, temp=temp,
                       outcomes=[StageResult(stage) for stage in STAGES])
    for outcome in run.outcomes:
        if outcome.stage.key not in selected:
            outcome.status = "not_selected"
    import signal

    previous = None
    if os.name != "nt":
        # An MCP cancel stops the benchmark with SIGTERM: write the report as for Ctrl+C.
        previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        run_stages(run, temp, echo=echo)
    except KeyboardInterrupt:
        run.interrupted = True
        say("\nThe benchmark was interrupted; writing the report of what ran.")
    except Exception as error:
        import traceback

        traceback.print_exc()
        run.error = f"{type(error).__name__}: {error}"
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        run.finished = datetime.now().astimezone()
        try:
            data = write_reports(run, folder)
            say()
            say(render(report_blocks(run), english).rstrip())
            say()
            say(f"The report is saved as {run.text_path}")
            say(f"Its data are saved as {run.json_path}")
            write_result(result_path, result_summary(data))
        finally:
            remove_temp(temp, lock)
    return run.exit_code


def main(argv=None):
    from utilities.Output_Streams import configure_output_streams

    configure_output_streams()
    parser = argparse.ArgumentParser(
        prog="Run_Benchmark.py",
        description="Time EMAP-SSN's heavy calculations on a fixed public sequence set and write a report.",
    )
    parser.add_argument("--stages", help="Stages to run, by number or name, comma-separated; "
                                         "the stages they need are added. All by default.")
    parser.add_argument("--result", help="Also write an English JSON summary to this file (MCP jobs use it).")
    parser.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--stage", help=argparse.SUPPRESS)
    parser.add_argument("--context", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.stage:
        return stage_main(arguments.stage, arguments.context)
    if arguments.limit is not None and arguments.limit < 2:
        parser.error("--limit must be at least 2")
    return run_benchmark(arguments.stages, arguments.result, arguments.limit)


if __name__ == "__main__":
    raise SystemExit(main())
