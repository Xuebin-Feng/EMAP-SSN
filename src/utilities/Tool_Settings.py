# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Shared JSON settings loading for command-line SSN tool execution."""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections.abc import MutableMapping
from pathlib import Path


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
    # Multiprocessing's spawn mode re-executes the tool module as
    # ``__mp_main__``. Propagate the selected file so spawned workers load the
    # same invocation snapshot instead of the GUI's mutable shared settings.
    os.environ["SSN_TOOL_SETTINGS_FILE"] = settings_path
    os.environ["SSN_TOOL_SETTINGS_SCRIPT"] = script_name
    return settings_path
