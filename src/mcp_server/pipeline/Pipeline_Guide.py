# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Planning knowledge for pipeline tools: inputs, outputs, next steps and workflows.

Pipeline_Guide.json holds the facts an agent needs to chain tools without
reading their source. Discovery imports nothing computational.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

from tools.tool_helpers.Tool_Pipeline import DEFAULT_DIRECTORY_PATHS, get_tool_spec

_GUIDE = json.loads(Path(__file__).with_name("Pipeline_Guide.json").read_text(encoding="utf-8"))
TOOL_GUIDE_FIELDS = ("stage", "purpose", "inputs", "outputs", "next", "requirements", "notes")


def tool_guide(tool_id):
    """One tool's planning entry, with every field present (empty when unused)."""
    entry = _GUIDE["tools"][get_tool_spec(tool_id).tool_id]
    empty = {"inputs": {}, "outputs": [], "next": [], "notes": [], "requirements": None}
    return {field: deepcopy(entry.get(field, empty.get(field))) for field in TOOL_GUIDE_FIELDS}


def path_rules():
    return list(_GUIDE["path_rules"])


def embedding_width(model):
    """Residue embedding width of a bundled model, or None for an unlisted plugin."""
    return _GUIDE["model_embedding_widths"].get(model)


def workflows():
    return deepcopy(_GUIDE["workflows"])


def _directory(document, key, project_root):
    value = (document.get("DIRECTORIES") or {}).get(key)
    if not isinstance(value, str) or not value.strip():
        value = DEFAULT_DIRECTORY_PATHS[key]
    return os.path.abspath(os.path.join(project_root, value))


def input_path(tool_id, field, document, project_root):
    """Resolve one input field the way its tool does, or None when it is unset."""
    spec = get_tool_spec(tool_id)
    value = (document.get(spec.settings_section) or {}).get(field)
    if not isinstance(value, str) or not value.strip():
        return None
    if os.path.isabs(value):
        return os.path.normpath(value)
    directory = _GUIDE["tools"][spec.tool_id]["inputs"][field]["directory"]
    return os.path.normpath(os.path.join(_directory(document, directory, project_root), value))


def watched_directories(tool_id, document, project_root):
    """Folders a tool writes to: its output directories and any input it writes beside."""
    spec = get_tool_spec(tool_id)
    folders = []
    for output in _GUIDE["tools"][spec.tool_id]["outputs"]:
        if "directory" in output:
            folder = _directory(document, output["directory"], project_root)
        else:
            beside = input_path(spec.tool_id, output["beside"], document, project_root)
            folder = os.path.dirname(beside) if beside else None
        if folder and os.path.normcase(folder) not in map(os.path.normcase, folders):
            folders.append(folder)
    return tuple(folders)


__all__ = ["embedding_width", "input_path", "path_rules", "tool_guide", "watched_directories", "workflows"]
