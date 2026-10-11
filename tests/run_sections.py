# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Run the test suite in sections, each section in its own process.

tests/sections.toml puts every tests/test_*.py module in one section (plus the
opt_vr suite as the section "opt_vr"). A section always runs as one process, the
same way in a partial run as in a full one, so its verdict does not depend on
which other sections ran. Sections run in parallel.

    python tests/run_sections.py --list               sections and what they cover
    python tests/run_sections.py layout caches        run the named sections
    python tests/run_sections.py --changed            sections that cover uncommitted changes
    python tests/run_sections.py --files src/X.py     sections that cover the given files
    python tests/run_sections.py --explain src/X.py   why a file maps to its sections
    python tests/run_sections.py --all                every section: the full suite

--changed and --files map each file to sections: a test module to its own
section; a file a test module (or one of its helper modules) imports or names
to that module's section; a file only other source files use to the sections of
its nearest users; plus the path rules in sections.toml. A partial run that
touches src/**/*.py also runs the repository-wide checks listed there.

Each section writes its output to its own log file; the summary names the logs
of failed sections and their failing tests. The exit code is 0 when every
selected section passed, 1 when any failed, and 2 for a usage error.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path("tests") / "sections.toml"
TEST_MODULE = re.compile(r"test\w*\.py")  # unittest discovery's default pattern
UNASSIGNED = "unassigned"
REPO_CHECKS = "repo-checks"
ROOT_PROCESS_MARKER = "EMAPSSN_TEST_ENVIRONMENT"  # set by tests/__init__.py
LOG_BASE = Path(tempfile.gettempdir()) / "emapssn-test-sections"
RUN_FOLDER = re.compile(r"\d{8}-\d{6}-\d+")
KEEP_RUNS = 10
DEFAULT_TIMEOUT = 1200
SKIP_DIRS = {".git", ".venv", "__pycache__", "player", "temp", "node_modules", ".claude"}
FILE_NAME = re.compile(
    r"[\w.\-]+\.(?:py|json|html|js|css|md|txt|sh|bat|cmd|ps1|toml|cff|ts|qm|qss|svg|"
    r"png|ico|ttf|otf|fasta|csv|yaml|yml)\b"
)
DOTTED_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
MODULE_LIKE = re.compile(r"[A-Z]\w*_\w+")


@dataclass
class Section:
    name: str
    description: str = ""
    modules: list = field(default_factory=list)
    tests: list = field(default_factory=list)
    seconds: float = 0.0
    cwd: str = ""
    discover: bool = False

    def is_vr(self):
        return self.discover


@dataclass
class Rule:
    paths: list
    sections: list


@dataclass
class Config:
    sections: dict
    rules: list
    full_suite: list
    repo_checks: list


@dataclass
class Selection:
    sections: set = field(default_factory=set)
    everything: bool = False
    repo_checks: bool = False
    reasons: dict = field(default_factory=lambda: defaultdict(list))
    unreferenced: list = field(default_factory=list)


@dataclass
class Result:
    name: str
    status: str
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    seconds: float = 0.0
    log: Path | None = None
    problems: list = field(default_factory=list)

    @property
    def failed(self):
        return self.status not in ("ok", "empty", "missing")


class UsageError(Exception):
    pass


# ----------------------------------------------------------------------------
# Configuration


def load_config(root=PROJECT_ROOT):
    data = tomllib.loads((Path(root) / CONFIG_PATH).read_text(encoding="utf-8"))
    sections = {}
    for name, entry in data.get("sections", {}).items():
        sections[name] = Section(
            name=name,
            description=entry.get("description", ""),
            modules=list(entry.get("modules", [])),
            seconds=float(entry.get("seconds", 0)),
            cwd=entry.get("cwd", ""),
            discover=bool(entry.get("discover", False)),
        )
    rules = [Rule(list(rule["paths"]), list(rule["sections"])) for rule in data.get("rules", [])]
    for rule in rules:
        for name in rule.sections:
            if name not in sections:
                raise UsageError(f"{CONFIG_PATH}: a rule names the unknown section {name!r}")
    return Config(
        sections=sections,
        rules=rules,
        full_suite=list(data.get("full_suite", {}).get("paths", [])),
        repo_checks=list(data.get("repo_checks", {}).get("tests", [])),
    )


def test_modules(root=PROJECT_ROOT):
    """Module names unittest discovery would collect from tests/ (top level)."""
    folder = Path(root) / "tests"
    return sorted(path.stem for path in folder.iterdir()
                  if path.is_file() and TEST_MODULE.fullmatch(path.name))


def check_assignment(config, modules):
    """Return (module -> section, problems) for the modules on disk."""
    owner, problems = {}, []
    present = set(modules)
    for section in config.sections.values():
        for module in section.modules:
            if module in owner:
                problems.append(f"{module} is listed in both {owner[module]} and {section.name}")
            elif module not in present:
                problems.append(f"{module} (section {section.name}) is not a test module in tests/")
            else:
                owner[module] = section.name
    for module in modules:
        if module not in owner:
            problems.append(f"{module} is in no section of {CONFIG_PATH.as_posix()}")
    return owner, problems


def runnable_sections(config, modules, root=PROJECT_ROOT):
    """The sections to run for a full suite: listed modules that exist, plus the leftovers."""
    owner, _ = check_assignment(config, modules)
    sections = {}
    for section in config.sections.values():
        if section.discover:
            sections[section.name] = section
            continue
        kept = [module for module in section.modules if owner.get(module) == section.name]
        sections[section.name] = Section(section.name, section.description, kept,
                                         seconds=section.seconds)
    leftovers = [module for module in modules if module not in owner]
    if leftovers:
        sections[UNASSIGNED] = Section(UNASSIGNED, "test modules missing from sections.toml",
                                       leftovers, seconds=5.0 * len(leftovers))
    return sections


# ----------------------------------------------------------------------------
# File references


def _glob_regex(pattern):
    out, index = [], 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            out.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            out.append(".*")
            index += 2
        elif pattern[index] == "*":
            out.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(out) + r"\Z")


def path_matches(path, patterns):
    return any(_glob_regex(pattern).match(path) for pattern in patterns)


def _git_lines(folder, *args):
    try:
        completed = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(folder), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()]


def _walk_files(folder):
    found = []
    for directory, subdirectories, names in os.walk(folder):
        subdirectories[:] = [name for name in subdirectories if name not in SKIP_DIRS]
        for name in names:
            found.append(Path(directory, name).relative_to(folder).as_posix())
    return found


def list_files(root):
    """Project files (tracked or untracked but not ignored), opt_vr included, as posix paths.

    Only names are listed; the index reads nothing but Python files, so git-ignored
    files such as keys never get opened."""
    root = Path(root)
    files = _git_lines(root, "ls-files", "--cached", "--others", "--exclude-standard")
    if files is None:
        files = _walk_files(root)
    files = [path for path in files if path != "opt_vr" and not path.startswith("opt_vr/")]
    vr = root / "opt_vr"
    if vr.is_dir():
        inner = _git_lines(vr, "ls-files", "--cached", "--others", "--exclude-standard") \
            if (vr / ".git").exists() else None
        files += ["opt_vr/" + path for path in (_walk_files(vr) if inner is None else inner)]
    return sorted(path for path in set(files)
                  if not any(part in SKIP_DIRS for part in path.split("/")))


def normalise(path):
    """A project-relative path as git prints it: forward slashes, no leading './'."""
    path = str(path).replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


class ReferenceIndex:
    """Which project files each Python file imports or names in a string."""

    def __init__(self, root=PROJECT_ROOT, files=None, extra=()):
        self.root = Path(root)
        self.files = sorted(set(files if files is not None else list_files(self.root)) | set(extra))
        self.by_basename = defaultdict(set)
        self.main_modules = defaultdict(set)
        self.vr_modules = defaultdict(set)
        for path in self.files:
            name = path.rsplit("/", 1)[-1]
            if name != "__init__.py":
                self.by_basename[name].add(path)
            if not path.endswith(".py"):
                continue
            parts = path[:-3].split("/")
            if parts[-1] == "__init__":
                parts = parts[:-1]
            index = self.vr_modules if path.startswith("opt_vr/") else self.main_modules
            for start in range(len(parts)):
                index[".".join(parts[start:])].add(path)
        self._refs = {}
        self._users = None

    # -- resolution

    def _resolve(self, dotted, in_vr):
        indexes = (self.vr_modules, self.main_modules) if in_vr else (self.main_modules,)
        pieces = dotted.split(".")
        for end in range(len(pieces), 0, -1):
            name = ".".join(pieces[:end])
            for index in indexes:
                if name in index:
                    return set(index[name])
        return set()

    def refs(self, path):
        """Files that the Python file at `path` imports or names (never itself)."""
        if path not in self._refs:
            self._refs[path] = self._scan(path)
        return self._refs[path]

    def _scan(self, path):
        found = set()
        try:
            source = (self.root / path).read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=path)
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            return found
        in_vr = path.startswith("opt_vr/")
        package = path.rsplit("/", 1)[0] if "/" in path else ""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found |= self._resolve(alias.name, in_vr)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.split("/") if package else []
                    base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                    stem = "/".join(base + (node.module.split(".") if node.module else []))
                    candidates = [stem + ".py", stem + "/__init__.py"]
                    candidates += [f"{stem}/{alias.name}.py" for alias in node.names]
                    found |= {candidate.lstrip("/") for candidate in candidates
                              if candidate.lstrip("/") in self._file_set}
                elif node.module:
                    found |= self._resolve(node.module, in_vr)
                    for alias in node.names:
                        found |= self._submodule(node.module, alias.name, in_vr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 400:
                value = node.value
                for name in FILE_NAME.findall(value):
                    found |= self.by_basename.get(name, set())
                if DOTTED_NAME.fullmatch(value) or MODULE_LIKE.fullmatch(value):
                    found |= self._resolve(value, in_vr)
        found.discard(path)
        return found

    def _submodule(self, package, name, in_vr):
        dotted = f"{package}.{name}"
        indexes = (self.vr_modules, self.main_modules) if in_vr else (self.main_modules,)
        for index in indexes:
            if dotted in index:
                return set(index[dotted])
        return set()

    @property
    def _file_set(self):
        if not hasattr(self, "_files_cache"):
            self._files_cache = set(self.files)
        return self._files_cache

    # -- reverse direction

    def users(self, path):
        """Python files that import or name `path`."""
        if self._users is None:
            users = defaultdict(set)
            for source in self.files:
                if source.endswith(".py"):
                    for target in self.refs(source):
                        users[target].add(source)
            self._users = users
        return self._users.get(path, set())


def _is_test_side(path):
    return path.startswith("tests/") or path.startswith("opt_vr/")


def module_references(index, module):
    """Files a main test module reaches: its own references and its helpers' (tests/*.py)."""
    start = f"tests/{module}.py"
    seen, stack, found = {start}, [start], set()
    while stack:
        for target in index.refs(stack.pop()):
            found.add(target)
            helper = (target.startswith("tests/") and target.endswith(".py")
                      and not TEST_MODULE.fullmatch(target.rsplit("/", 1)[-1])
                      and target not in ("tests/__init__.py", "tests/run_sections.py"))
            if helper and target not in seen:
                seen.add(target)
                stack.append(target)
    return found


# ----------------------------------------------------------------------------
# Selection


def changed_files(root=PROJECT_ROOT, base="HEAD"):
    """Files that differ from `base` (default: uncommitted), untracked ones included."""
    root = Path(root)
    changed = _git_lines(root, "diff", "--name-only", base, "--")
    if changed is None:
        raise UsageError(f"git cannot compare {root} with {base!r}")
    changed += _git_lines(root, "ls-files", "--others", "--exclude-standard") or []
    vr = root / "opt_vr"
    if (vr / ".git").exists():
        inner = (_git_lines(vr, "diff", "--name-only", "HEAD", "--") or []) + \
                (_git_lines(vr, "ls-files", "--others", "--exclude-standard") or [])
        changed += ["opt_vr/" + path for path in inner]
    return sorted(set(changed))


class Selector:
    def __init__(self, config, root=PROJECT_ROOT, index=None, modules=None):
        self.config = config
        self.root = Path(root)
        self.modules = modules if modules is not None else test_modules(self.root)
        self.owner, _ = check_assignment(config, self.modules)
        self.index = index if index is not None else ReferenceIndex(self.root)
        self._by_target = None
        self._vr_targets = None

    def section_of(self, module):
        return self.owner.get(module, UNASSIGNED)

    def _targets(self):
        if self._by_target is None:
            by_target = defaultdict(set)
            for module in self.modules:
                for target in module_references(self.index, module):
                    by_target[target].add(module)
            vr_targets = set()
            for path in self.index.files:
                if path.startswith("opt_vr/") and path.endswith(".py"):
                    vr_targets |= self.index.refs(path)
            self._by_target, self._vr_targets = by_target, vr_targets
        return self._by_target, self._vr_targets

    def _direct(self, path):
        """Sections whose tests reach `path` directly, with a reason per section."""
        by_target, vr_targets = self._targets()
        hits = defaultdict(list)
        for module in sorted(by_target.get(path, ())):
            hits[self.section_of(module)].append(module)
        if path in vr_targets and not path.startswith("opt_vr/") and "opt_vr" in self.config.sections:
            hits["opt_vr"].append("opt_vr code")
        return hits

    def explain(self, path):
        """(sections, reasons) for one file; sections None means every section."""
        path = normalise(path)
        if path_matches(path, self.config.full_suite):
            return None, [f"{path} is a full-suite file in {CONFIG_PATH.as_posix()}"]
        name = path.rsplit("/", 1)[-1]
        if path.startswith("tests/") and "/" not in path[6:] and TEST_MODULE.fullmatch(name):
            section = self.section_of(name[:-3])
            return {section}, [f"{path} is a test module of section {section}"]
        sections, reasons = set(), []
        for rule in self.config.rules:
            if path_matches(path, rule.paths):
                sections |= set(rule.sections)
                reasons.append(f"rule {', '.join(rule.paths)} -> {', '.join(rule.sections)}")
        for section, modules in sorted(self._direct(path).items()):
            sections.add(section)
            reasons.append(f"{section}: used directly by {', '.join(modules)}")
        if not sections and not _is_test_side(path):
            frontier, seen = {path}, {path}
            while frontier and not sections:
                frontier = {user for item in frontier for user in self.index.users(item)
                            if user not in seen and not _is_test_side(user)}
                seen |= frontier
                for user in sorted(frontier):
                    for section, modules in sorted(self._direct(user).items()):
                        sections.add(section)
                        reasons.append(f"{section}: through {user} ({', '.join(modules)})")
        if not sections:
            reasons.append("no test module reaches it")
        return sections, reasons

    def select(self, paths):
        selection = Selection()
        for path in map(normalise, paths):
            sections, reasons = self.explain(path)
            selection.reasons[path] = reasons
            if sections is None:
                selection.everything = True
                continue
            selection.sections |= sections
            if not sections:
                selection.unreferenced.append(path)
            if path.startswith("src/") and path.endswith(".py"):
                selection.repo_checks = True
        return selection

    def pending_repo_checks(self, selected):
        """Repository-wide checks whose own section is not selected."""
        return [test for test in self.config.repo_checks
                if self.section_of(test.split(".")[1]) not in selected]


# ----------------------------------------------------------------------------
# Running


SUMMARY_RAN = re.compile(rb"^Ran (\d+) tests? in ([\d.]+)s\s*$", re.M)
SUMMARY_STATUS = re.compile(rb"^(OK|FAILED|NO TESTS RAN)(?: \(([^)]*)\))?\s*$", re.M)
PROBLEM_LINE = re.compile(rb"^(FAIL|ERROR|UNEXPECTED SUCCESS): (.+?)\s*$", re.M)


def parse_log(data):
    """Return (status, counts, problems) from a unittest log (bytes)."""
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    problems = [f"{kind.decode()}: {name.decode('utf-8', 'replace')}"
                for kind, name in PROBLEM_LINE.findall(data)]
    ran = SUMMARY_RAN.findall(data)
    statuses = SUMMARY_STATUS.findall(data)
    if not ran or not statuses:
        return "crashed", counts, problems
    counts["tests"] = int(ran[-1][0])
    word, details = statuses[-1]
    for item in details.decode("ascii", "replace").split(","):
        key, _, value = item.strip().partition("=")
        if key in counts and value.isdigit():
            counts[key] = int(value)
        elif key == "unexpected successes" and value.isdigit():
            counts["failures"] += int(value)
    status = {b"OK": "ok", b"FAILED": "failed", b"NO TESTS RAN": "empty"}[word]
    if status == "ok" and counts["tests"] == 0:
        status = "empty"
    return status, counts, problems


def child_environment(environ=None):
    """The runner's environment without the marker that would make a child share
    the runner's private test folders: each section sets up its own."""
    environ = dict(os.environ if environ is None else environ)
    environ.pop(ROOT_PROCESS_MARKER, None)
    return environ


def section_command(section, python=None, verbose=False, patterns=()):
    python = python or sys.executable
    flags = (["-v"] if verbose else []) + [item for pattern in patterns for item in ("-k", pattern)]
    if section.discover:
        return [python, "-m", "unittest", "discover", *flags, "-s", "tests", "-t", "."]
    names = list(section.tests) or [f"tests.{module}" for module in section.modules]
    return [python, "-m", "unittest", *flags, *names]


def _kill_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                       capture_output=True, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired, PermissionError):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


class Runner:
    def __init__(self, root, log_dir, jobs, timeout, python=None, verbose=False, patterns=(),
                 out=None):
        self.root = Path(root)
        self.log_dir = Path(log_dir)
        self.jobs = max(1, jobs)
        self.timeout = timeout
        self.python = python
        self.verbose = verbose
        self.patterns = list(patterns)
        self.out = out or sys.stdout
        self._lock = threading.Lock()
        self._processes = set()
        self._stopping = False

    def _say(self, text):
        with self._lock:
            print(text, file=self.out, flush=True)

    def run_one(self, section):
        cwd = self.root / section.cwd if section.cwd else self.root
        log = self.log_dir / f"{section.name}.log"
        if section.discover and not (cwd / "tests").is_dir():
            self._say(f"  skip  {section.name:<12} {cwd / 'tests'} is not here")
            return Result(section.name, "missing", log=None)
        command = section_command(section, self.python, self.verbose, self.patterns)
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" \
            else {"start_new_session": True}
        start = time.monotonic()
        with open(log, "wb") as handle:
            handle.write((" ".join(command) + "\n\n").encode("utf-8", "replace"))
            handle.flush()
            if self._stopping:
                return Result(section.name, "interrupted", log=log)
            process = subprocess.Popen(command, cwd=cwd, env=child_environment(),
                                       stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT, **options)
            with self._lock:
                self._processes.add(process)
            try:
                process.wait(timeout=self.timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                _kill_tree(process)
                timed_out = True
            finally:
                with self._lock:
                    self._processes.discard(process)
        seconds = time.monotonic() - start
        status, counts, problems = parse_log(log.read_bytes())
        if timed_out:
            status = "timeout"
        elif self._stopping and status == "crashed":
            status = "interrupted"
        elif status == "ok" and process.returncode not in (0, None):
            status = "crashed"
        result = Result(section.name, status, seconds=seconds, log=log, problems=problems, **counts)
        mark = {"ok": "pass", "empty": "none"}.get(status, status.upper())
        self._say(f"  {mark:<5} {section.name:<12} {result.tests:>5} tests {seconds:7.1f} s")
        return result

    def run(self, sections):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        results = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.jobs)
        futures = [executor.submit(self.run_one, section) for section in sections]
        pending = set(futures)
        try:
            while pending:
                # A timed wait, so Ctrl+C reaches this thread on Windows too.
                done, pending = concurrent.futures.wait(
                    pending, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED)
                results.extend(future.result() for future in done)
        except KeyboardInterrupt:
            self._stopping = True
            for future in futures:
                future.cancel()
            with self._lock:
                running = list(self._processes)
            for process in running:
                _kill_tree(process)
            executor.shutdown(wait=True)
            raise
        executor.shutdown(wait=True)
        order = {section.name: index for index, section in enumerate(sections)}
        return sorted(results, key=lambda result: order[result.name])


# ----------------------------------------------------------------------------
# Durations and logs


def _durations_path():
    return LOG_BASE / "durations.json"


def load_durations():
    try:
        return {key: float(value) for key, value in
                json.loads(_durations_path().read_text(encoding="utf-8")).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def save_durations(results, full_sections):
    durations = load_durations()
    for result in results:
        if result.status in ("ok", "failed") and result.name in full_sections:
            durations[result.name] = round(result.seconds, 1)
    try:
        LOG_BASE.mkdir(parents=True, exist_ok=True)
        _durations_path().write_text(json.dumps(durations, indent=1, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def new_log_dir():
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = LOG_BASE / f"{stamp}-{os.getpid()}"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        runs = sorted((path for path in LOG_BASE.iterdir()
                       if path.is_dir() and RUN_FOLDER.fullmatch(path.name)),
                      key=lambda path: path.stat().st_mtime)
        for old in runs[:-KEEP_RUNS]:
            shutil.rmtree(old, ignore_errors=True)
    except OSError:
        pass
    return folder


def expected_seconds(section, durations):
    return durations.get(section.name) or section.seconds or 5.0 * max(1, len(section.modules))


# ----------------------------------------------------------------------------
# Command line


def default_jobs():
    return max(2, min(8, (os.cpu_count() or 4) // 3))


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        prog="run_sections.py", description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring (tests/run_sections.py) for how files map to sections.")
    parser.add_argument("sections", nargs="*", help="section names to run")
    parser.add_argument("--all", action="store_true", help="run every section (the full suite)")
    parser.add_argument("--changed", action="store_true",
                        help="add the sections that cover files changed since --base")
    parser.add_argument("--base", default="HEAD",
                        help="commit --changed compares the working tree with (default HEAD)")
    parser.add_argument("--repo", type=Path,
                        help="git checkout --changed reads (default: this project); lets a "
                             "snapshot copy run the sections a working tree's changes need")
    parser.add_argument("--files", nargs="+", default=[], metavar="FILE",
                        help="add the sections that cover these files (project-relative paths)")
    parser.add_argument("--skip", nargs="+", default=[], metavar="SECTION",
                        help="leave these sections out")
    parser.add_argument("--list", action="store_true", help="list the sections and exit")
    parser.add_argument("--explain", nargs="+", metavar="FILE",
                        help="show which sections cover each file, and why, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would run without running it")
    parser.add_argument("-j", "--jobs", type=int, default=default_jobs(),
                        help="sections run at once (default %(default)s)")
    parser.add_argument("-k", dest="patterns", action="append", default=[],
                        help="only tests matching this unittest -k pattern (repeatable)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose unittest output in logs")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="seconds before a section is stopped (default %(default)s)")
    parser.add_argument("--log-dir", type=Path, help="folder for the section logs")
    return parser.parse_args(argv)


def _relative(path, root):
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            return candidate.as_posix()
    return normalise(path)


def print_list(config, modules, out):
    durations = load_durations()
    _, problems = check_assignment(config, modules)
    for section in config.sections.values():
        size = "opt_vr suite" if section.discover else f"{len(section.modules)} modules"
        print(f"  {section.name:<12} {size:<13} ~{expected_seconds(section, durations):4.0f} s  "
              f"{section.description}", file=out)
    for problem in problems:
        print(f"  warning: {problem}", file=out)


def main(argv=None, root=PROJECT_ROOT, out=None):
    out = out or sys.stdout
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    root = Path(root)
    try:
        config = load_config(root)
    except (OSError, tomllib.TOMLDecodeError, UsageError, KeyError) as error:
        print(f"error: {CONFIG_PATH.as_posix()}: {error}", file=out)
        return 2
    modules = test_modules(root)
    sections = runnable_sections(config, modules, root)
    owner, problems = check_assignment(config, modules)
    if args.list:
        print_list(config, modules, out)
        return 0

    selector_root = args.repo or root
    if args.explain:
        selector = Selector(config, root=selector_root)
        for path in args.explain:
            relative = _relative(path, selector_root)
            chosen, reasons = selector.explain(relative)
            label = "every section" if chosen is None else (", ".join(sorted(chosen)) or "none")
            print(f"{relative}: {label}", file=out)
            for reason in reasons:
                print(f"    {reason}", file=out)
        return 0

    unknown = [name for name in args.sections + args.skip if name not in sections]
    if unknown:
        print(f"error: unknown section(s): {', '.join(unknown)}; choose from: "
              f"{', '.join(sections)}", file=out)
        return 2
    if not (args.all or args.sections or args.changed or args.files):
        print("Nothing selected. Name sections, or use --changed, --files or --all.\n", file=out)
        print_list(config, modules, out)
        return 2
    for problem in problems:
        print(f"warning: {problem}", file=out)

    chosen = set(args.sections)
    repo_checks = []
    if args.all:
        chosen |= set(sections)
    if args.changed or args.files:
        selector = Selector(config, root=selector_root)
        paths = [_relative(path, selector_root) for path in args.files]
        if args.changed:
            try:
                paths += changed_files(selector_root, args.base)
            except UsageError as error:
                print(f"error: {error}", file=out)
                return 2
        selection = selector.select(sorted(set(paths)))
        print(f"{len(paths)} file(s) considered:", file=out)
        for path, reasons in selection.reasons.items():
            print(f"  {path}", file=out)
            for reason in reasons:
                print(f"      {reason}", file=out)
        chosen |= set(sections) if selection.everything else selection.sections & set(sections)
        if selection.repo_checks and not args.all:
            repo_checks = selector.pending_repo_checks(chosen)
    chosen -= set(args.skip)
    to_run = [sections[name] for name in sections if name in chosen]
    if repo_checks:
        to_run.append(Section(REPO_CHECKS, "repository-wide checks", tests=repo_checks, seconds=10))
    if not to_run:
        print("No section covers these files; nothing to run.", file=out)
        return 0

    durations = load_durations()
    to_run.sort(key=lambda section: -expected_seconds(section, durations))
    names = ", ".join(section.name for section in to_run)
    print(f"Running {len(to_run)} section(s), {min(args.jobs, len(to_run))} at a time: {names}",
          file=out)
    if args.dry_run:
        for section in to_run:
            cwd = section.cwd or "."
            print(f"  [{cwd}] {' '.join(section_command(section, verbose=args.verbose, patterns=args.patterns))}",
                  file=out)
        return 0

    log_dir = args.log_dir or new_log_dir()
    runner = Runner(root, log_dir, args.jobs, args.timeout, verbose=args.verbose,
                    patterns=args.patterns, out=out)
    start = time.monotonic()
    try:
        results = runner.run(to_run)
    except KeyboardInterrupt:
        print(f"\nInterrupted; logs so far are in {log_dir}", file=out)
        return 130
    wall = time.monotonic() - start
    if not args.patterns:
        save_durations(results, set(config.sections))
    return report(results, wall, log_dir, out)


def report(results, wall, log_dir, out):
    print("", file=out)
    print(f"  {'section':<12} {'tests':>6} {'skip':>5} {'fail':>5} {'error':>5} {'seconds':>8}  result",
          file=out)
    for result in results:
        print(f"  {result.name:<12} {result.tests:>6} {result.skipped:>5} {result.failures:>5} "
              f"{result.errors:>5} {result.seconds:>8.1f}  {result.status}", file=out)
    failed = [result for result in results if result.failed]
    total = sum(result.tests for result in results)
    busy = sum(result.seconds for result in results)
    print(f"\n{total} tests in {wall:.1f} s ({busy:.1f} s of section time); "
          f"{len(failed)} of {len(results)} section(s) failed. Logs: {log_dir}", file=out)
    for result in failed:
        print(f"\n{result.name}: {result.status}, log {result.log}", file=out)
        for problem in result.problems[:25]:
            print(f"    {problem}", file=out)
        if len(result.problems) > 25:
            print(f"    ... {len(result.problems) - 25} more in the log", file=out)
    if results and all(result.status in ("empty", "missing") for result in results):
        print("No tests ran.", file=out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
