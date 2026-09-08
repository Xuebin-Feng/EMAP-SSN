# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Consolidated standalone SSN tool directories, settings, process execution, and search benchmarks."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time
import uuid

# =====================================================================
# 1. Canonical Directory Defaults & Registry
# =====================================================================

DEFAULT_DIRECTORY_PATHS = {
    "EMBED_DIR": os.path.join("Embeddings"),
    "FASTA_DIR": os.path.join("Input_Files", "Sequence_Sets"),
    "MSA_DIR": os.path.join("Input_Files", "Multiple_Alignments"),
    "NETWORK_DIR": os.path.join("Input_Files", "Networks_EValues"),
    "REPORT_DIR": os.path.join("Analysis_Results", "Alignment_Report"),
    "SETTING_EXPORT_DIR": os.path.join("Cache_Files", "Exported_Settings"),
}

TOOL_DIRECTORY_KEYS = {
    "Align_Similarity_Matrix.py": ("EMBED_DIR", "NETWORK_DIR"),
    "Align_Substitution_Matrix.py": ("FASTA_DIR", "NETWORK_DIR"),
    "Embedding_Cropping.py": ("FASTA_DIR", "EMBED_DIR"),
    "Embedding_Extraction.py": ("FASTA_DIR", "EMBED_DIR"),
    "Embedding_Injection.py": ("FASTA_DIR", "EMBED_DIR"),
    "Embedding_MSA.py": ("FASTA_DIR", "EMBED_DIR", "NETWORK_DIR", "MSA_DIR"),
    "Embedding_PWA.py": ("EMBED_DIR", "REPORT_DIR"),
    "Embedding_SSEARCH.py": ("EMBED_DIR", "REPORT_DIR"),
    "Generate_Embeddings.py": ("FASTA_DIR", "EMBED_DIR"),
    "Network_Extraction.py": ("FASTA_DIR", "NETWORK_DIR"),
    "Network_Injection.py": ("EMBED_DIR", "NETWORK_DIR"),
    "Parse_BLAST_Output.py": ("FASTA_DIR", "NETWORK_DIR"),
    "Sanitize_Sequences.py": ("FASTA_DIR",),
    "Sparse_MSA_Converter.py": ("MSA_DIR",),
}


def project_directory_defaults(project_root):
    """Return canonical default directories anchored to ``project_root``."""
    return {
        key: os.path.normpath(os.path.join(project_root, relative_path))
        for key, relative_path in DEFAULT_DIRECTORY_PATHS.items()
    }


def fill_missing_directory_defaults(settings):
    """Fill absent or blank global directories without replacing custom values."""
    if not isinstance(settings, MutableMapping):
        raise TypeError("Tool settings must be a mutable mapping.")

    directories = settings.get("DIRECTORIES")
    if not isinstance(directories, MutableMapping):
        directories = {}
        settings["DIRECTORIES"] = directories

    for key, default_path in DEFAULT_DIRECTORY_PATHS.items():
        value = directories.get(key)
        if value is None or not str(value).strip():
            directories[key] = default_path

    return settings


# =====================================================================
# 2. Tool Settings Management & Parsing
# =====================================================================

class ToolSettingsError(ValueError):
    """Raised when an explicitly supplied tool settings file is unusable."""


def default_settings_path(project_root):
    return os.path.join(project_root, "tools_settings.json")


def inherited_settings_path(script_path):
    """Return a spawn-inherited settings path only for its originating tool."""
    if os.environ.get("SSN_TOOL_SETTINGS_SCRIPT") != os.path.basename(script_path):
        return None
    return os.environ.get("SSN_TOOL_SETTINGS_FILE") or None


def _settings_parser(script_name):
    parser = argparse.ArgumentParser(
        prog=script_name,
        description=(
            "Run this SSN tool with an exported JSON settings file. When the "
            "argument is omitted, the project-root tools_settings.json is used."
        ),
    )
    parser.add_argument(
        "settings_json",
        nargs="?",
        help="JSON file exported for this tool by EMAPSSN_Tools.py",
    )
    return parser


def select_settings_path(script_name, project_root, argv=None):
    """Return ``(path, explicit)`` for a tool's optional positional argument."""
    parser = _settings_parser(script_name)
    args = parser.parse_args(argv)
    if args.settings_json:
        return os.path.abspath(os.fspath(args.settings_json)), True
    return default_settings_path(project_root), False


def read_settings_document(settings_path, script_name, *, explicit):
    """Read and validate a shared or exported tool settings document."""
    path = Path(settings_path)
    if not path.is_file():
        if explicit:
            raise ToolSettingsError(f"Settings file was not found: {path}")
        return {}

    try:
        with path.open("r", encoding="utf-8") as settings_handle:
            document = json.load(settings_handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        if explicit:
            raise ToolSettingsError(
                f"Could not read settings file '{path}': {error}"
            ) from error
        print(f"Failed to load user settings: {error}")
        return {}

    return validate_settings_document(
        document,
        script_name,
        explicit=explicit,
        source_label=f"Settings file '{path}'",
    )


def validate_settings_document(
    document,
    script_name,
    *,
    explicit=True,
    source_label="Settings document",
):
    """Validate and copy an in-memory tool settings document."""
    if not isinstance(document, MutableMapping):
        message = f"{source_label} must contain a JSON object."
        if explicit:
            raise ToolSettingsError(message)
        print(f"Failed to load user settings: {message}")
        return {}

    directories = document.get("DIRECTORIES", {})
    tool_settings = document.get(script_name)
    if explicit:
        if "DIRECTORIES" not in document or not isinstance(
            directories, MutableMapping
        ):
            raise ToolSettingsError(
                f"{source_label} must contain a DIRECTORIES object."
            )
        if not isinstance(tool_settings, MutableMapping):
            raise ToolSettingsError(
                f"{source_label} does not contain the required "
                f"'{script_name}' object. Export settings for this tool and try again."
            )
    else:
        if not isinstance(directories, MutableMapping):
            directories = {}
        if not isinstance(tool_settings, MutableMapping):
            tool_settings = {}

    return {
        "DIRECTORIES": dict(directories),
        script_name: dict(tool_settings or {}),
    }


def _has_value(value):
    return value is not None and str(value).strip() != ""


def _coerce_setting(value, original):
    if isinstance(original, bool):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "on", "yes", "1"}:
                return True
            if normalized in {"false", "off", "no", "0"}:
                return False
        return value
    if isinstance(original, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if isinstance(original, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if isinstance(original, list) and isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    if original is None and isinstance(value, str):
        if value == "None":
            return None
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value
    return value


def apply_settings_document(namespace, document, script_name, project_root):
    """Apply recognized settings to a tool module namespace."""
    for key, value in document.get("DIRECTORIES", {}).items():
        if key not in namespace or not _has_value(value):
            continue
        resolved = os.fspath(value)
        if not os.path.isabs(resolved):
            resolved = os.path.normpath(os.path.join(project_root, resolved))
        namespace[key] = resolved

    for key, value in document.get(script_name, {}).items():
        if key not in namespace or not _has_value(value):
            continue
        coerced = _coerce_setting(value, namespace[key])
        if (
            isinstance(coerced, str)
            and key.endswith("_DIR")
            and not os.path.isabs(coerced)
        ):
            coerced = os.path.normpath(os.path.join(project_root, coerced))
        namespace[key] = coerced


def load_tool_settings(namespace, script_path, project_root, argv=None):
    """Parse, load, and apply settings for one direct tool invocation."""
    script_name = os.path.basename(script_path)
    parser = _settings_parser(script_name)
    args = parser.parse_args(argv)
    explicit = bool(args.settings_json)
    settings_path = (
        os.path.abspath(os.fspath(args.settings_json))
        if explicit
        else default_settings_path(project_root)
    )
    try:
        document = read_settings_document(
            settings_path,
            script_name,
            explicit=explicit,
        )
    except ToolSettingsError as error:
        parser.error(str(error))
    apply_settings_document(namespace, document, script_name, project_root)
    os.environ["SSN_TOOL_SETTINGS_FILE"] = settings_path
    os.environ["SSN_TOOL_SETTINGS_SCRIPT"] = script_name
    return settings_path


# =====================================================================
# 3. Tool Execution Preparation & Specifications
# =====================================================================

@dataclass(frozen=True)
class ToolSpec:
    """One allowlisted standalone tool and its settings contract."""

    tool_id: str
    script_name: str
    settings_section: str
    required_directories: tuple[str, ...]
    output_directories: tuple[str, ...]
    relative_script: str

    def script_path(self, project_root):
        return os.path.abspath(os.path.join(project_root, self.relative_script))


@dataclass(frozen=True)
class ToolInvocation:
    """A fully resolved command that a caller may execute or display."""

    tool: ToolSpec
    argv: tuple[str, ...]
    cwd: str
    settings_path: str
    owns_settings_snapshot: bool = False


_TOOL_IDS = {
    "Align_Similarity_Matrix.py": "align_similarity_matrix",
    "Align_Substitution_Matrix.py": "align_substitution_matrix",
    "Embedding_Cropping.py": "embedding_cropping",
    "Embedding_Extraction.py": "embedding_extraction",
    "Embedding_Injection.py": "embedding_injection",
    "Embedding_MSA.py": "embedding_msa",
    "Embedding_PWA.py": "embedding_pwa",
    "Embedding_SSEARCH.py": "embedding_ssearch",
    "Generate_Embeddings.py": "generate_embeddings",
    "Network_Extraction.py": "network_extraction",
    "Network_Injection.py": "network_injection",
    "Parse_BLAST_Output.py": "parse_blast_output",
    "Sanitize_Sequences.py": "sanitize_sequences",
    "Sparse_MSA_Converter.py": "sparse_msa_converter",
}

_TOOL_OUTPUT_DIRECTORIES = {
    "Align_Similarity_Matrix.py": ("NETWORK_DIR",),
    "Align_Substitution_Matrix.py": ("NETWORK_DIR",),
    "Embedding_Cropping.py": ("EMBED_DIR",),
    "Embedding_Extraction.py": ("EMBED_DIR",),
    "Embedding_Injection.py": ("EMBED_DIR",),
    "Embedding_MSA.py": ("MSA_DIR",),
    "Embedding_PWA.py": ("REPORT_DIR",),
    "Embedding_SSEARCH.py": ("REPORT_DIR",),
    "Generate_Embeddings.py": ("EMBED_DIR",),
    "Network_Extraction.py": ("NETWORK_DIR",),
    "Network_Injection.py": ("NETWORK_DIR",),
    "Parse_BLAST_Output.py": ("NETWORK_DIR",),
    "Sanitize_Sequences.py": ("FASTA_DIR",),
    "Sparse_MSA_Converter.py": ("MSA_DIR",),
}

TOOL_SPECS = tuple(
    ToolSpec(
        tool_id=tool_id,
        script_name=script_name,
        settings_section=script_name,
        required_directories=tuple(TOOL_DIRECTORY_KEYS[script_name]),
        output_directories=tuple(_TOOL_OUTPUT_DIRECTORIES[script_name]),
        relative_script=os.path.join("src", "tools", script_name),
    )
    for script_name, tool_id in _TOOL_IDS.items()
)
_SPECS_BY_ID = {spec.tool_id: spec for spec in TOOL_SPECS}
_SPECS_BY_SCRIPT = {spec.script_name: spec for spec in TOOL_SPECS}


def list_tool_specs():
    """Return the stable allowlisted tool catalog."""
    return TOOL_SPECS


def get_tool_spec(tool_id):
    """Resolve one stable ID without accepting arbitrary script paths."""
    try:
        return _SPECS_BY_ID[str(tool_id)]
    except KeyError as error:
        raise KeyError(f"Unknown SSN tool ID: {tool_id}") from error


def get_tool_spec_for_script(script_path):
    """Resolve the allowlisted tool represented by ``script_path``."""
    script_name = os.path.basename(os.fspath(script_path))
    try:
        return _SPECS_BY_SCRIPT[script_name]
    except KeyError as error:
        raise KeyError(f"Unknown SSN tool script: {script_name}") from error


def build_settings_document(spec, directories, tool_settings):
    """Build and validate the existing exported JSON document shape."""
    if not isinstance(spec, ToolSpec):
        spec = get_tool_spec(spec)
    directory_values = directories if isinstance(directories, Mapping) else {}
    document = {
        "DIRECTORIES": {
            key: directory_values.get(key, "")
            for key in spec.required_directories
        },
        spec.settings_section: dict(tool_settings),
    }
    return validate_settings_document(
        document,
        spec.settings_section,
        explicit=True,
    )


def write_json_document(path, document, *, atomic=True, trailing_newline=False):
    """Write one JSON document without emitting terminal output."""
    target = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    write_path = f"{target}.{os.getpid()}.{uuid.uuid4().hex}.partial" if atomic else target
    try:
        with open(write_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=4)
            if trailing_newline:
                handle.write("\n")
        if atomic:
            os.replace(write_path, target)
    finally:
        if atomic and os.path.exists(write_path):
            os.unlink(write_path)
    return target


def load_shared_settings(project_root):
    """Load the GUI's shared document while preserving its fallback behavior."""
    settings_path = default_settings_path(project_root)
    document = {}
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            document = dict(loaded) if isinstance(loaded, Mapping) else loaded
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    fill_missing_directory_defaults(document)
    return document


def save_shared_tool_settings(
    project_root,
    spec,
    tool_settings,
    *,
    base_document=None,
):
    """Replace one GUI tool section and preserve all unrelated sections."""
    if not isinstance(spec, ToolSpec):
        spec = get_tool_spec(spec)
    document = (
        load_shared_settings(project_root)
        if base_document is None
        else dict(base_document)
    )
    fill_missing_directory_defaults(document)
    document[spec.settings_section] = dict(tool_settings)
    settings_path = default_settings_path(project_root)
    write_json_document(settings_path, document, atomic=False)
    return settings_path


def _normalized_source_document(spec, settings_source):
    if isinstance(settings_source, Mapping):
        return validate_settings_document(
            settings_source,
            spec.settings_section,
            explicit=True,
        )
    source_path = os.path.abspath(os.fspath(settings_source))
    return read_settings_document(
        source_path,
        spec.settings_section,
        explicit=True,
    )


def create_settings_snapshot(spec, settings_source, *, snapshot_directory=None):
    """Copy a mapping or exported file into a validated immutable snapshot."""
    if not isinstance(spec, ToolSpec):
        spec = get_tool_spec(spec)
    document = _normalized_source_document(spec, settings_source)
    snapshot_root = os.path.abspath(
        os.fspath(
            snapshot_directory
            or os.path.join(tempfile.gettempdir(), "ssn_tool_invocations")
        )
    )
    os.makedirs(snapshot_root, mode=0o700, exist_ok=True)
    snapshot_path = os.path.join(
        snapshot_root,
        f"{spec.tool_id}-{uuid.uuid4().hex}.json",
    )
    write_json_document(snapshot_path, document, atomic=True, trailing_newline=True)
    try:
        os.chmod(snapshot_path, 0o600)
    except OSError:
        pass
    return snapshot_path


def resolve_tool_directories(
    spec,
    settings_source,
    project_root,
    *,
    output_only=False,
):
    """Resolve effective configured directories without changing the document."""
    if not isinstance(spec, ToolSpec):
        spec = get_tool_spec(spec)
    document = _normalized_source_document(spec, settings_source)
    configured = document.get("DIRECTORIES", {})
    keys = spec.output_directories if output_only else spec.required_directories
    resolved = {}
    for key in keys:
        value = configured.get(key)
        if value is None or not str(value).strip():
            value = DEFAULT_DIRECTORY_PATHS[key]
        path = os.fspath(value)
        if not os.path.isabs(path):
            path = os.path.join(project_root, path)
        resolved[key] = os.path.abspath(os.path.normpath(path))
    return resolved


def prepare_gui_invocation(script_path, project_root, *, python_executable=None):
    """Resolve the GUI's unchanged visible-terminal invocation."""
    spec = get_tool_spec_for_script(script_path)
    resolved_script = spec.script_path(project_root)
    return ToolInvocation(
        tool=spec,
        argv=(python_executable or sys.executable, "-u", resolved_script),
        cwd=os.path.dirname(resolved_script),
        settings_path=default_settings_path(project_root),
        owns_settings_snapshot=False,
    )


def prepare_headless_invocation(
    tool_id,
    settings_source,
    project_root,
    *,
    python_executable=None,
    snapshot_directory=None,
):
    """Prepare an explicit-settings subprocess invocation for a future adapter."""
    spec = get_tool_spec(tool_id)
    snapshot_path = create_settings_snapshot(
        spec,
        settings_source,
        snapshot_directory=snapshot_directory,
    )
    resolved_script = spec.script_path(project_root)
    return ToolInvocation(
        tool=spec,
        argv=(
            python_executable or sys.executable,
            "-u",
            resolved_script,
            snapshot_path,
        ),
        cwd=os.path.dirname(resolved_script),
        settings_path=snapshot_path,
        owns_settings_snapshot=True,
    )


def prepare_exported_invocation(
    tool_id,
    settings_path,
    project_root,
    *,
    python_executable=None,
):
    """Resolve an invocation that uses an existing exported settings file."""
    spec = get_tool_spec(tool_id)
    resolved_settings = os.path.abspath(os.fspath(settings_path))
    read_settings_document(
        resolved_settings,
        spec.settings_section,
        explicit=True,
    )
    resolved_script = spec.script_path(project_root)
    return ToolInvocation(
        tool=spec,
        argv=(
            python_executable or sys.executable,
            "-u",
            resolved_script,
            resolved_settings,
        ),
        cwd=os.path.dirname(resolved_script),
        settings_path=resolved_settings,
        owns_settings_snapshot=False,
    )


def format_invocation_command(invocation):
    """Format an invocation for the GUI's informational command preview."""
    executable, *arguments = invocation.argv
    rendered = [f'"{executable}"']
    rendered.extend(
        argument if argument == "-u" else f'"{argument}"'
        for argument in arguments
    )
    return " ".join(rendered)


# =====================================================================
# 4. SSEARCH Benchmark & Timing Selection
# =====================================================================

class SearchTiming:
    """Optional runner hook; start/stop bracket drained processing only."""
    def __init__(self):
        self.created = time.perf_counter()
        self.setup = self.processing = self.shutdown = 0.0

    def start(self):
        self.started = time.perf_counter()
        self.setup = self.started - self.created

    def stop(self):
        self.stopped = time.perf_counter()
        self.processing = self.stopped - self.started

    def finish(self):
        self.shutdown = time.perf_counter() - self.stopped

    def warm_executor(self, executor, count, callback=None):
        """Materialize every lazy executor thread before processing timing."""
        barrier = threading.Barrier(count)

        def warm():
            try:
                if callback is not None:
                    callback()
                barrier.wait(timeout=30)
            except BaseException:
                barrier.abort()
                raise

        futures = [executor.submit(warm) for _ in range(count)]
        for future in futures:
            future.result()


@dataclass
class SearchPlan:
    candidate: object
    variant: str
    precision: str
    lanes: int = 1
    observations: list = field(default_factory=list)
    results: dict = field(default_factory=dict)

    @property
    def key(self):
        return (self.candidate.spec, self.variant, self.precision, self.lanes)

    @property
    def execution(self):
        return (self.candidate, self.variant, self.precision, self.lanes)

    @property
    def label(self):
        return f"{self.candidate.display_name} / {self.variant} / {self.precision} / {self.lanes} lane(s)"

    def predicted(self, costs):
        remaining = sum(cost for index, cost in costs.items() if index not in self.results)
        if not remaining:
            return 0.0
        if not self.observations:
            return math.inf
        return statistics.median(
            timing.setup + timing.shutdown + rate * remaining
            for timing, rate in self.observations
        )


def stratified(tasks, lengths, count):
    ordered = sorted(tasks, key=lambda task: (lengths[int(task[0])], int(task[0])))
    count = min(len(ordered), max(0, int(count)))
    if count < 2:
        return ordered[:count]
    return [ordered[i * (len(ordered) - 1) // (count - 1)] for i in range(count)]


def rank_plans(plans, costs):
    remaining = [plan for plan in plans if plan.observations]
    ranked = []
    while remaining:
        best = min(plan.predicted(costs) for plan in remaining)
        eligible = [plan for plan in remaining if plan.predicted(costs) <= best * 1.05]
        winner = min(eligible, key=lambda plan: (
            plan.variant != "serial", plan.lanes, plan.variant == "tiled",
        ))
        ranked.append(winner)
        remaining.remove(winner)
    return ranked


class SearchSelector:
    def __init__(self, tasks, lengths, query_length, execute, clock=None):
        self.tasks, self.lengths = tasks, lengths
        self.costs = {int(task[0]): max(1, query_length * lengths[int(task[0])]) for task in tasks}
        self.execute, self.clock = execute, clock or time.perf_counter
        self.started = self.clock()
        self.excluded = 0.0
        self.budget = 5.0
        self.plans = []
        self.messages = []
        self.sampled_ids = set()
        self.sample = stratified(tasks, lengths, 16)

    @property
    def elapsed(self):
        return max(0.0, self.clock() - self.started - self.excluded)

    def ranked(self):
        return rank_plans(self.plans, self.costs)

    def skip(self, plan, reason):
        message = f"{plan.label}: {reason}"
        self.messages.append(message)
        print(f"[Hardware] {message}")

    def allowed(self, plan):
        ranked = self.ranked()
        if self.elapsed >= self.budget:
            self.skip(plan, ("repeat skipped" if plan.observations else "unmeasured") + ": tuning budget exhausted")
            return False
        if not ranked:
            return True
        observations = plan.observations or [
            obs for other in self.plans if other.candidate.spec == plan.candidate.spec
            for obs in other.observations
        ] or ranked[0].observations
        work = sum(self.costs[int(task[0])] for task in self.sample)
        estimate = statistics.median(t.setup + t.shutdown + rate * work for t, rate in observations)
        if plan.variant == "pool" and not plan.observations:
            estimate = max(estimate, 1.0 + statistics.median(rate * work for _, rate in observations))
        if estimate > self.budget - self.elapsed or ranked[0].predicted(self.costs) <= 2 * estimate:
            self.skip(plan, ("repeat skipped" if plan.observations else "unmeasured") + ": estimated trial cost cannot repay tuning")
            return False
        return True

    def trial(self, plan, sample=None, required=False):
        if not required and not self.allowed(plan):
            return False
        sample = self.sample if sample is None else sample
        timing = SearchTiming()
        try:
            payload = self.execute(plan, sample, timing)
            expected = {int(task[0]) for task in sample}
            received = {int(row["index"]): row for row in payload}
            if len(payload) != len(received) or received.keys() != expected:
                raise ValueError("Search trial returned missing, duplicate, or unexpected targets")
            if not math.isfinite(timing.processing) or timing.processing < 0:
                raise ValueError("Invalid search timing")
            work = sum(self.costs[index] for index in expected)
            plan.observations.append((timing, max(timing.processing, 1e-9) / max(work, 1)))
            plan.results.update(received)
            self.sampled_ids.update(received)
            if plan not in self.plans:
                self.plans.append(plan)
            print(f"[Hardware] {plan.label}: setup={timing.setup:.3f}s; "
                  f"processing={timing.processing:.3f}s; shutdown={timing.shutdown:.3f}s; "
                  f"estimated remaining={plan.predicted(self.costs):.3f}s")
            return True
        except Exception as error:
            self.skip(plan, f"failed: {type(error).__name__}: {error}")
            return False

    def baseline(self, plan):
        if not self.trial(plan, stratified(self.tasks, self.lengths, 4), required=True):
            return False
        duration = plan.predicted(self.costs)
        timing, rate = plan.observations[0]
        total_duration = timing.setup + timing.shutdown + rate * sum(self.costs.values())
        self.budget = min(5.0, max(0.5, 0.05 * total_duration))
        mean_cost = sum(self.costs.values()) / len(self.costs)
        rate = plan.observations[0][1]
        count = min(64, max(16, math.ceil(0.1 / max(rate * mean_cost, 1e-9))))
        self.sample = stratified(self.tasks, self.lengths, count)
        return duration <= 1.0


__all__ = [
    "DEFAULT_DIRECTORY_PATHS",
    "TOOL_DIRECTORY_KEYS",
    "project_directory_defaults",
    "fill_missing_directory_defaults",
    "ToolSettingsError",
    "default_settings_path",
    "inherited_settings_path",
    "select_settings_path",
    "read_settings_document",
    "validate_settings_document",
    "apply_settings_document",
    "load_tool_settings",
    "ToolSpec",
    "ToolInvocation",
    "TOOL_SPECS",
    "list_tool_specs",
    "get_tool_spec",
    "get_tool_spec_for_script",
    "build_settings_document",
    "write_json_document",
    "load_shared_settings",
    "save_shared_tool_settings",
    "create_settings_snapshot",
    "resolve_tool_directories",
    "prepare_gui_invocation",
    "prepare_headless_invocation",
    "prepare_exported_invocation",
    "format_invocation_command",
    "SearchTiming",
    "SearchPlan",
    "stratified",
    "rank_plans",
    "SearchSelector",
]
