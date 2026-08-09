# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
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
USAGE_TERM_FIELDS = frozenset({
    "source_url",
    "license_id",
    "license_url",
    "restriction",
    "requires_acknowledgement",
})


def _read_literal_assignments(filepath, names):
    with open(filepath, "r", encoding="utf-8") as source_file:
        tree = ast.parse(source_file.read(), filename=filepath)

    values = {}
    for item in tree.body:
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if isinstance(target, ast.Name) and target.id in names:
                values[target.id] = ast.literal_eval(item.value)
    return values


def read_plugin_metadata(filepath):
    """Read declarative plugin metadata without importing model dependencies."""
    values = _read_literal_assignments(
        filepath,
        {"SUPPORTED_MODELS", "MODEL_EXECUTION_MODES"},
    )

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


def validate_model_usage_terms(supported, usage_terms):
    """Validate optional declarative licensing metadata for external models."""
    if usage_terms is None:
        return {}
    if not isinstance(usage_terms, dict):
        raise ValueError("MODEL_USAGE_TERMS must be a literal mapping.")

    unknown_models = sorted(set(usage_terms) - set(supported))
    if unknown_models:
        raise ValueError(
            "MODEL_USAGE_TERMS contains unsupported model(s): "
            f"{unknown_models}."
        )
    for model_name, terms in usage_terms.items():
        if not isinstance(terms, dict) or set(terms) != USAGE_TERM_FIELDS:
            raise ValueError(
                f"MODEL_USAGE_TERMS['{model_name}'] must contain exactly "
                f"{sorted(USAGE_TERM_FIELDS)}."
            )
        for field in USAGE_TERM_FIELDS - {"requires_acknowledgement"}:
            if not isinstance(terms[field], str) or not terms[field].strip():
                raise ValueError(
                    f"MODEL_USAGE_TERMS['{model_name}']['{field}'] must be "
                    "a non-empty string."
                )
        if not isinstance(terms["requires_acknowledgement"], bool):
            raise ValueError(
                f"MODEL_USAGE_TERMS['{model_name}'] acknowledgement flag "
                "must be boolean."
            )
    return usage_terms


def read_model_usage_terms(filepath):
    """Read optional model usage terms statically without importing a plugin."""
    supported, _ = read_plugin_metadata(filepath)
    values = _read_literal_assignments(filepath, {"MODEL_USAGE_TERMS"})
    return validate_model_usage_terms(
        supported,
        values.get("MODEL_USAGE_TERMS"),
    )


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


def discover_model_usage_terms(plugin_dir):
    """Return all declared model usage terms, rejecting duplicate models."""
    discovered = {}
    seen_models = set()
    for filepath in sorted(glob.glob(os.path.join(plugin_dir, "*.py"))):
        if os.path.basename(filepath) == "__init__.py":
            continue
        supported, _ = read_plugin_metadata(filepath)
        duplicates = sorted(seen_models.intersection(supported))
        if duplicates:
            raise ValueError(f"Model(s) declared by multiple plugins: {duplicates}.")
        seen_models.update(supported)
        discovered.update(read_model_usage_terms(filepath))
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
    validate_model_usage_terms(
        supported,
        getattr(plugin, "MODEL_USAGE_TERMS", None),
    )
    mode = modes[model_name]
    if mode not in ALLOWED_EXECUTION_MODES:
        raise ValueError(
            f"Plugin declares unsupported execution mode '{mode}' for '{model_name}'."
        )
    return mode
