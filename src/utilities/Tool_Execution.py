# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral preparation for standalone SSN tool execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
import sys
import tempfile
import uuid

from utilities.Tool_Directories import (
    TOOL_DIRECTORY_KEYS,
    fill_missing_directory_defaults,
)
from utilities.Tool_Settings import (
    ToolSettingsError,
    default_settings_path,
    read_settings_document,
    validate_settings_document,
)


@dataclass(frozen=True)
class ToolSpec:
    """One allowlisted standalone tool and its settings contract."""

    tool_id: str
    script_name: str
    settings_section: str
    required_directories: tuple[str, ...]
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

if set(_TOOL_IDS) != set(TOOL_DIRECTORY_KEYS):
    raise RuntimeError("The tool execution catalog does not match Tool_Directories.")

TOOL_SPECS = tuple(
    ToolSpec(
        tool_id=tool_id,
        script_name=script_name,
        settings_section=script_name,
        required_directories=tuple(TOOL_DIRECTORY_KEYS[script_name]),
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


__all__ = [
    "TOOL_SPECS",
    "ToolInvocation",
    "ToolSettingsError",
    "ToolSpec",
    "build_settings_document",
    "create_settings_snapshot",
    "format_invocation_command",
    "get_tool_spec",
    "get_tool_spec_for_script",
    "list_tool_specs",
    "load_shared_settings",
    "prepare_gui_invocation",
    "prepare_exported_invocation",
    "prepare_headless_invocation",
    "save_shared_tool_settings",
    "write_json_document",
]
