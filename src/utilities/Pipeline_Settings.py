# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Discoverable MCP contracts and export-only serialization (no tool imports)."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path

from jsonschema import Draft202012Validator

from utilities.PLM_Plugin_Utils import discover_model_execution_modes
from utilities.FASTA_Sanitization import sanitize_sequence
from utilities.Tool_Directories import DEFAULT_DIRECTORY_PATHS
from utilities.Tool_Execution import get_tool_spec, list_tool_specs


SCHEMA_VERSION = 1
_CONTRACTS = json.loads(Path(__file__).with_name("Pipeline_Settings_Contracts.json").read_text(encoding="utf-8"))
DESCRIPTIONS = {
    "sanitize_sequences": "Clean, deduplicate, and optionally filter FASTA sequences.",
    "generate_embeddings": "Generate residue embeddings from FASTA sequences.",
    "embedding_cropping": "Crop embedding residues using a corresponding FASTA set.",
    "embedding_extraction": "Extract embeddings for a FASTA subset.",
    "embedding_injection": "Add FASTA sequences to an embedding database.",
    "align_similarity_matrix": "Build a network from embedding alignment scores.",
    "align_substitution_matrix": "Build a BLAST substitution-matrix network.",
    "parse_blast_output": "Import external BLAST tabular results into a network.",
    "embedding_msa": "Build a multiple alignment from embeddings and a network.",
    "sparse_msa_converter": "Convert aligned FASTA files to sparse MSA storage.",
    "network_injection": "Extend a network using an expanded embedding database.",
    "network_extraction": "Extract a network for a FASTA subset.",
    "embedding_pwa": "Align two stored or manually supplied protein sequences.",
    "embedding_ssearch": "Search an embedding database with a stored or manual query.",
}


class PipelineSettingsError(ValueError):
    """One or more field-specific configuration errors."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))


def _required_fields(*names):
    return {"required": list(names), "properties": {
        name: {"type": "string", "pattern": r"\S"} for name in names
    }}


def _when(key, value, *required):
    return {"if": {"properties": {key: {"const": value}}, "required": [key]},
            "then": _required_fields(*required)}


def _parameter_schema(tool_id, project_root, *, execution=True):
    spec = get_tool_spec(tool_id)
    contract = deepcopy(_CONTRACTS[spec.script_name])
    props = contract["properties"]
    for prop in props.values():
        if prop.get("x-choice-provider") == "plm_models":
            models = list(discover_model_execution_modes(os.path.join(project_root, "src", "resources", "pLM_models")))
            prop["enum"] = models + ([None] if prop["default"] is None else [])
        if not execution and "enum" in prop:
            # Empty selectors are valid incomplete exports, not executable inputs.
            prop["enum"] += [""]
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema",
              "type": "object", "additionalProperties": False, "properties": props}
    if not execution:
        return schema
    rules = [_required_fields(*contract["required"])] if contract["required"] else []
    if tool_id == "embedding_msa":
        rules.append(_when("USE_SEQUENCE_FILTER", True, "INPUT_FASTA"))
    elif tool_id == "sparse_msa_converter":
        rules.append(_when("CONVERT_ALL", False, "INPUT_FASTA"))
    elif tool_id == "embedding_pwa":
        for prefix in ("REF", "TAR"):
            rules.extend([_when(f"MANUAL_{prefix}_SEQ", True, f"{prefix}_SEQUENCE"),
                          _when(f"MANUAL_{prefix}_SEQ", False, "INPUT_EMBED")])
        rules.append({"if": {"properties": {"MANUAL_REF_SEQ": {"const": True}, "MANUAL_TAR_SEQ": {"const": True}},
                              "required": ["MANUAL_REF_SEQ", "MANUAL_TAR_SEQ"]},
                      "then": _required_fields("EMBEDDING_MODEL")})
    elif tool_id == "embedding_ssearch":
        rules.extend([_when("MANUAL_QUERY_SEQ", True, "QUERY_SEQUENCE"),
                      _when("MANUAL_QUERY_SEQ", False, "QUERY_HEADER")])
    if rules:
        schema["allOf"] = rules
    return schema


def get_pipeline_schema(tool_id, project_root):
    spec = get_tool_spec(tool_id)
    schema = _parameter_schema(tool_id, project_root)
    example = {key: ("esmc_300m" if key == "MODEL_NAME" else "example.fasta" if "FASTA" in key else "example.h5")
               for key in _CONTRACTS[spec.script_name]["required"]}
    if tool_id == "sparse_msa_converter":
        example = {"CONVERT_ALL": True}
    elif tool_id == "embedding_pwa":
        example = {"MANUAL_REF_SEQ": True, "REF_SEQUENCE": "ACDE", "MANUAL_TAR_SEQ": True, "TAR_SEQUENCE": "ACDF"}
    elif tool_id == "parse_blast_output":
        example["INPUT_BLAST_TABULAR"] = "example.tsv"
    return {"schema_version": SCHEMA_VERSION, "tool_id": tool_id,
            "description": DESCRIPTIONS[tool_id], "parameters_schema": schema,
            "directories_schema": {"type": "object", "additionalProperties": False, "properties": {
                key: {"type": ["string", "null"], "default": DEFAULT_DIRECTORY_PATHS[key],
                      "description": "Project-relative or absolute directory; omitted, null, or blank uses the project default."}
                for key in spec.required_directories}},
            "headless_behavior": {"SHOW_REGRESSION_PLOT": False} if tool_id == "embedding_msa" else
                ({"length_distribution": "50-bin text table; no figure"} if tool_id == "sanitize_sequences" else {}),
            "validation_notes": ["Defaults are applied before conditional requirements.",
                "Enabled minimum/maximum length bounds must be ordered; custom BLAST columns must be distinct.",
                "Tiled execution cannot select CPU; explicit TF32 cannot select a non-CUDA device.",
                "Configuration validation does not check file existence, credentials, or hardware readiness.",
                "Relative directories resolve against the project root; input filenames retain each tool's directory semantics."],
            "example": {"tool_id": tool_id, "parameters": example}}


def _errors(schema, values, prefix):
    errors = []
    for error in Draft202012Validator(schema).iter_errors(values):
        field = ".".join([prefix, *map(str, error.absolute_path)])
        if error.validator == "additionalProperties" and isinstance(error.instance, dict):
            for key in error.instance.keys() - error.schema.get("properties", {}).keys():
                errors.append({"field": f"{field}.{key}", "message": "Unknown setting key."})
        elif error.validator == "required":
            for key in error.validator_value:
                if key not in error.instance:
                    errors.append({"field": f"{field}.{key}", "message": "Required input is missing."})
        else:
            errors.append({"field": field, "message": error.message})
    def finite(value, path):
        if isinstance(value, float) and not math.isfinite(value):
            errors.append({"field": path, "message": "Must be a finite JSON number."})
        elif isinstance(value, dict):
            for key, child in value.items(): finite(child, f"{path}.{key}")
        elif isinstance(value, list):
            for i, child in enumerate(value): finite(child, f"{path}.{i}")
    finite(values, prefix)
    return errors


def _combination_errors(tool_id, values):
    errors = []
    def fail(field, message): errors.append({"field": f"parameters.{field}", "message": message})
    if tool_id == "sanitize_sequences" and values["ENABLE_LENGTH_FILTER"]:
        low, high = values["MIN_SEQ_LENGTH"], values["MAX_SEQ_LENGTH"]
        if low > 0 and high > 0 and low > high:
            fail("MAX_SEQ_LENGTH", "Must be at least MIN_SEQ_LENGTH when both bounds are enabled.")
    if tool_id == "parse_blast_output" and values["BLAST_LAYOUT"] == "custom_columns":
        if len({values[k] for k in ("QUERY_COLUMN", "SUBJECT_COLUMN", "EVALUE_COLUMN")}) != 3:
            fail("EVALUE_COLUMN", "Query, subject, and E-value columns must be distinct.")
    if values.get("ACCELERATOR_PRECISION") == "tf32" and values.get("DEVICE_SELECTION") not in (None, "auto"):
        if not values["DEVICE_SELECTION"].startswith("cuda:"):
            fail("ACCELERATOR_PRECISION", "TF32 requires auto selection or a NVIDIA CUDA device; availability is checked at execution.")
    if values.get("EXECUTION_MODE") == "tiled" and values.get("DEVICE_SELECTION") == "cpu":
        fail("EXECUTION_MODE", "Tiled execution requires an accelerator, not CPU.")
    for switch, field in (("MANUAL_REF_SEQ", "REF_SEQUENCE"), ("MANUAL_TAR_SEQ", "TAR_SEQUENCE"),
                          ("MANUAL_QUERY_SEQ", "QUERY_SEQUENCE")):
        if values.get(switch) and not sanitize_sequence(values[field])[0]:
            fail(field, "Manual sequence is empty after sanitization.")
    return errors


def normalize_pipeline_settings(tool_id, project_root, *, parameters=None, directories=None,
                                settings_document=None, settings_path=None):
    """Return a side-effect-free preview; submission must use this exact document."""
    result = {"schema_version": SCHEMA_VERSION, "tool_id": tool_id, "valid": False,
              "errors": [], "settings_document": None, "effective_directories": {},
              "applied_defaults": {}, "overrides": []}
    def fail(field, message): result["errors"].append({"field": field, "message": message})
    try:
        spec = get_tool_spec(tool_id)
    except KeyError as error:
        fail("tool_id", str(error)); return result
    if sum(v is not None for v in (parameters, settings_document, settings_path)) != 1:
        fail("input", "Provide exactly one of parameters, settings_document, or settings_path.")
    if directories is not None and parameters is None:
        fail("directories", "Allowed only with parameters.")
    if result["errors"]:
        return result
    if parameters is not None:
        document = {"DIRECTORIES": {} if directories is None else directories, spec.settings_section: parameters}
    else:
        document = settings_document
        if settings_path is not None:
            try:
                path = os.fspath(settings_path)
                if not os.path.isabs(path):
                    path = os.path.join(project_root, path)
                with open(path, encoding="utf-8") as handle:
                    document = json.load(handle)
            except (OSError, ValueError, TypeError) as error:
                fail("settings_path", str(error))
                return result
    if not isinstance(document, dict):
        fail("settings_document", "Must be a JSON object.")
        return result
    allowed_sections = {s.settings_section for s in list_tool_specs()} | {"DIRECTORIES"}
    for key in document.keys() - allowed_sections:
        fail(str(key), "Unknown settings section.")
    for section in ("DIRECTORIES", spec.settings_section):
        if not isinstance(document.get(section), dict):
            fail(section, "A JSON object is required.")
    if result["errors"]:
        return result
    directory_values = document["DIRECTORIES"]
    allowed_dirs = spec.required_directories if parameters is not None else DEFAULT_DIRECTORY_PATHS
    for key, value in directory_values.items():
        if key not in allowed_dirs:
            fail(f"directories.{key}", "Unknown or inapplicable directory key.")
        elif value is not None and not isinstance(value, str):
            fail(f"directories.{key}", "Must be a string or null.")
    if result["errors"]:
        return result
    resolved = {}
    for key in spec.required_directories:
        value = directory_values.get(key)
        if value is None or not value.strip():
            value = DEFAULT_DIRECTORY_PATHS[key]
            result["applied_defaults"][f"directories.{key}"] = value
        if not os.path.isabs(value):
            value = os.path.join(project_root, value)
        resolved[key] = os.path.abspath(os.path.normpath(value))
    schema = _parameter_schema(tool_id, project_root)
    supplied = document[spec.settings_section]
    values = {key: deepcopy(prop["default"]) for key, prop in schema["properties"].items()}
    result["applied_defaults"].update({f"parameters.{key}": deepcopy(value) for key, value in values.items() if key not in supplied})
    values.update(supplied)
    result["errors"].extend(_errors(schema, values, "parameters"))
    if not result["errors"]:
        result["errors"].extend(_combination_errors(tool_id, values))
    if result["errors"]:
        return result
    # Match Tool_Settings' treatment of configured tool-specific directories.
    for key, value in values.items():
        if key.endswith("_DIR") and isinstance(value, str) and value.strip():
            if not os.path.isabs(value):
                value = os.path.join(project_root, value)
            values[key] = os.path.abspath(os.path.normpath(value))
    if tool_id == "embedding_msa" and values["SHOW_REGRESSION_PLOT"]:
        result["overrides"].append({"field": "parameters.SHOW_REGRESSION_PLOT", "supplied": True, "effective": False, "reason": "Headless execution disables figures."})
        values["SHOW_REGRESSION_PLOT"] = False
    result.update(valid=True, effective_directories=resolved,
                  settings_document={"DIRECTORIES": resolved, spec.settings_section: values})
    return result


def serialize_export_settings(tool_id, values, project_root):
    """Convert GUI text controls only; do not alter GUI execution or require inputs."""
    schema = _parameter_schema(tool_id, project_root, execution=False)
    output = deepcopy(values)
    for key, value in output.items():
        prop = schema["properties"].get(key)
        if prop is None or not isinstance(value, str):
            continue
        types = prop.get("type", [])
        types = [types] if isinstance(types, str) else types
        if key == "HOST_CACHE_GB":
            types = ["number"]
        if value.strip() in ("", "None", "null") and "null" in types and "string" not in types:
            output[key] = None
        elif "integer" in types or "number" in types or "boolean" in types or "array" in types:
            if key == "HOST_CACHE_GB" and value == "auto":
                continue
            try:
                output[key] = json.loads(value)
            except (ValueError, TypeError):
                # Leave the invalid value for a field-specific validation error.
                pass
    errors = _errors(schema, output, "parameters")
    if errors:
        raise PipelineSettingsError(errors)
    return output
