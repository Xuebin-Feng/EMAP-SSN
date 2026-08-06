# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static discovery and validation for protein-language-model plugins."""

from __future__ import annotations

import ast
import glob
import os


ALLOWED_EXECUTION_MODES = frozenset({"local", "remote_api"})


def read_plugin_metadata(filepath):
    """Read declarative plugin metadata without importing model dependencies."""
    with open(filepath, "r", encoding="utf-8") as source_file:
        tree = ast.parse(source_file.read(), filename=filepath)

    values = {}
    for item in tree.body:
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if isinstance(target, ast.Name) and target.id in {
                "SUPPORTED_MODELS",
                "MODEL_EXECUTION_MODES",
            }:
                values[target.id] = ast.literal_eval(item.value)

    supported = values.get("SUPPORTED_MODELS")
    modes = values.get("MODEL_EXECUTION_MODES")
    if not isinstance(supported, list) or not all(
        isinstance(model, str) and model for model in supported
    ):
        raise ValueError("SUPPORTED_MODELS must be a literal list of model names.")
    if not isinstance(modes, dict):
        raise ValueError(
            "MODEL_EXECUTION_MODES must be a literal mapping for every supported model."
        )
    if set(modes) != set(supported):
        missing = sorted(set(supported) - set(modes))
        extra = sorted(set(modes) - set(supported))
        raise ValueError(
            "MODEL_EXECUTION_MODES must exactly cover SUPPORTED_MODELS "
            f"(missing={missing}, extra={extra})."
        )
    invalid = {
        model: mode for model, mode in modes.items()
        if mode not in ALLOWED_EXECUTION_MODES
    }
    if invalid:
        raise ValueError(f"Unknown model execution mode(s): {invalid}.")
    return supported, modes


def discover_model_execution_modes(plugin_dir):
    """Return all declared model execution modes, rejecting duplicates."""
    discovered = {}
    for filepath in sorted(glob.glob(os.path.join(plugin_dir, "*.py"))):
        if os.path.basename(filepath) == "__init__.py":
            continue
        supported, modes = read_plugin_metadata(filepath)
        for model in supported:
            if model in discovered:
                raise ValueError(f"Model '{model}' is declared by multiple plugins.")
            discovered[model] = modes[model]
    return discovered


def validate_loaded_plugin(plugin, model_name):
    """Validate imported metadata and return the selected model's mode."""
    supported = getattr(plugin, "SUPPORTED_MODELS", None)
    modes = getattr(plugin, "MODEL_EXECUTION_MODES", None)
    if not isinstance(supported, list) or not isinstance(modes, dict):
        raise ValueError(
            "pLM plugin is missing SUPPORTED_MODELS or MODEL_EXECUTION_MODES."
        )
    if set(modes) != set(supported):
        raise ValueError(
            "pLM plugin MODEL_EXECUTION_MODES must exactly cover SUPPORTED_MODELS."
        )
    if model_name not in supported:
        raise ValueError(f"Plugin does not support model '{model_name}'.")
    mode = modes[model_name]
    if mode not in ALLOWED_EXECUTION_MODES:
        raise ValueError(
            f"Plugin declares unsupported execution mode '{mode}' for '{model_name}'."
        )
    return mode
