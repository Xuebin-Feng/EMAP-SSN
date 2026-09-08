# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Consolidated protein language model plugin discovery, execution modes, and licensing."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import tempfile

# =====================================================================
# 1. Declarative Plugin Discovery & Metadata Validation
# =====================================================================

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


# =====================================================================
# 2. Model License Verification & Acceptance Storage
# =====================================================================

PROJECT_ROOT = os.path.abspath(str(Path(__file__).resolve().parents[3]))
DEFAULT_ACCEPTANCE_FILE = os.path.join(
    PROJECT_ROOT,
    "src",
    "resources",
    "pLM_models",
    "ankh_license.json",
)


class ModelLicenseAcceptanceRequired(PermissionError):
    """Raised before model access when required terms have not been accepted."""


def model_terms_fingerprint(model_name, terms):
    """Return a stable fingerprint that changes with the model's declared terms."""
    payload = {
        "schema": 1,
        "model_name": model_name,
        "terms": terms,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_acceptance_store(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def is_model_license_accepted(model_name, terms, path=DEFAULT_ACCEPTANCE_FILE):
    """Return true only for an exact, current model-and-terms acknowledgement."""
    if not terms or not terms.get("requires_acknowledgement", False):
        return True
    record = _read_acceptance_store(path).get(model_name)
    return bool(
        isinstance(record, dict)
        and record.get("terms_fingerprint")
        == model_terms_fingerprint(model_name, terms)
    )


def record_model_license_acceptance(
    model_name,
    terms,
    path=DEFAULT_ACCEPTANCE_FILE,
):
    """Atomically record acceptance without collecting identity or other PII."""
    if not terms or not terms.get("requires_acknowledgement", False):
        return

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    data = _read_acceptance_store(path)
    data[model_name] = {
        "license_id": terms["license_id"],
        "source_url": terms["source_url"],
        "license_url": terms["license_url"],
        "terms_fingerprint": model_terms_fingerprint(model_name, terms),
        "accepted_at": datetime.now(timezone.utc).isoformat(),
    }

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=directory,
            prefix=".model-license-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def format_model_usage_terms(model_name, terms):
    """Create the common human-readable notice used by GUI and CLI paths."""
    return (
        f"Model: {model_name}\n"
        f"Weights license: {terms['license_id']}\n"
        f"Restriction: {terms['restriction']}\n"
        f"Model source: {terms['source_url']}\n"
        f"License information: {terms['license_url']}\n\n"
        "The EMAP-SSN integration code is Apache-2.0, but the separately "
        "downloaded model weights are not."
    )


def format_model_selector_label(model_name, terms):
    """Label separately licensed weights without changing the model identifier."""
    if not terms:
        return model_name
    restriction = terms["restriction"].lower()
    if "non-commercial" in restriction:
        return f"{model_name} [non-commercial]"
    return f"{model_name} [separate terms]"


def prompt_for_model_license_acceptance(
    model_name,
    terms,
    path=DEFAULT_ACCEPTANCE_FILE,
    input_func=input,
    output_func=print,
):
    """Review terms and optionally persist an exact terminal acknowledgement."""
    output_func(format_model_usage_terms(model_name, terms))
    response = input_func(
        "\nType I ACCEPT to record acceptance, or press Enter to cancel: "
    )
    if response.strip() != "I ACCEPT":
        return False
    record_model_license_acceptance(model_name, terms, path)
    return True


def require_model_license_acceptance(
    model_name,
    terms,
    path=DEFAULT_ACCEPTANCE_FILE,
):
    """Fail closed before model loading if current terms need acknowledgement."""
    if is_model_license_accepted(model_name, terms, path):
        return
    notice = format_model_usage_terms(model_name, terms)
    raise ModelLicenseAcceptanceRequired(
        notice
        + "\n\nNo model files were accessed. To review and accept these terms "
        "from a terminal, run:\n"
        f"  python src/tools/Generate_Embeddings.py "
        f"--accept-model-license {model_name}"
    )


__all__ = [
    "ALLOWED_EXECUTION_MODES",
    "USAGE_TERM_FIELDS",
    "read_plugin_metadata",
    "validate_model_usage_terms",
    "read_model_usage_terms",
    "discover_model_execution_modes",
    "discover_model_usage_terms",
    "validate_loaded_plugin",
    "PROJECT_ROOT",
    "DEFAULT_ACCEPTANCE_FILE",
    "ModelLicenseAcceptanceRequired",
    "model_terms_fingerprint",
    "is_model_license_accepted",
    "record_model_license_acceptance",
    "format_model_usage_terms",
    "format_model_selector_label",
    "prompt_for_model_license_acceptance",
    "require_model_license_acceptance",
]
