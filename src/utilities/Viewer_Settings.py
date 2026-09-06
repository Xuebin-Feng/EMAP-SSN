# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Explicit Viewer configuration shared by MCP, CLI and the configuration GUI.

No GUI imports or personal-settings reads are allowed in this module. Paths in
documents are portable; normalized snapshots contain absolute paths.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path

from utilities.Viewer_Defaults import (
    INPUT_PROFILE_DEFAULTS, VISUAL_PROFILE_DEFAULTS, PHYSICS_PROFILE_DEFAULTS,
    DIRECTORY_PROFILE_DEFAULTS, PROFILE_ENUM_VALUES, PROFILE_RANGES,
)


class ViewerSettingsError(ValueError):
    pass


DEFAULTS = {
    **DIRECTORY_PROFILE_DEFAULTS, **INPUT_PROFILE_DEFAULTS,
    **VISUAL_PROFILE_DEFAULTS, **PHYSICS_PROFILE_DEFAULTS,
    "TARGET_CACHE_PATH": "", "TARGET_CACHE_MODE": "existing",
    "SAVED_CONFIG_DIR": "$cache_file$/Saved_Config", "BOX_SCALE": 2.0,
    "PACKING_PADDING": 10.0, "MAX_FORCE_LIMIT": 20.0,
    "MAX_TOTAL_REPULSION_FORCE": 0.0,
}
REQUIRED = ("TARGET_CACHE_PATH", "NODE_FASTA_FILE", "INPUT_HDF5", "MSA_FILE", "UMAP_MODE")
ALIASES = {"$input_file$": "INPUT_FILE_DIR", "$cache_file$": "CACHE_FILE_DIR",
           "$analysis_result$": "ANALYSIS_RESULT_DIR"}
FILES = {"NODE_FASTA_FILE": "FASTA_DIR", "INPUT_HDF5": "HDF5_DIR",
         "MSA_FILE": "MSA_DIR", "TARGET_CACHE_PATH": "SAVED_LAYOUT_DIR"}


def get_viewer_settings_schema():
    properties = {}
    for key, default in DEFAULTS.items():
        kind = ("boolean" if isinstance(default, bool) else "integer" if isinstance(default, int)
                else "number" if isinstance(default, float) else "string")
        properties[key] = {"type": ["number", "null"] if default is None else kind,
                           "default": default}
        if key in PROFILE_ENUM_VALUES:
            properties[key]["enum"] = sorted(PROFILE_ENUM_VALUES[key])
        if key in PROFILE_RANGES:
            properties[key].update(zip(("minimum", "maximum"), PROFILE_RANGES[key]))
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": list(REQUIRED), "allOf": [
                {"if": {"properties": {"UMAP_MODE": {"const": True}}},
                 "then": {"required": ["UMAP_NEIGHBORS"]},
                 "else": {"required": ["SIMILARITY_THRESHOLD", "TOP_EDGE_PERCENT"]}},
            ], "description": "Complete Viewer settings. Alignment networks also require ALIGNMENT_SCORE and NORM_MODE. MSA_FILE='' disables alignment. Scientific settings must match the cache manifest. Relative file selectors use their configured directory; paths with directories use project root."}


def read_viewer_settings(*, settings_document=None, settings_path=None, project_root):
    if (settings_document is None) == (settings_path is None):
        raise ViewerSettingsError("Supply exactly one of settings_document or settings_path.")
    if settings_path is not None:
        path = Path(settings_path)
        if not path.is_absolute():
            path = Path(project_root) / path
        try:
            with path.open(encoding="utf-8") as handle:
                settings_document = json.load(handle)
        except (OSError, ValueError) as error:
            raise ViewerSettingsError(f"settings_path: {error}") from error
    if not isinstance(settings_document, dict):
        raise ViewerSettingsError("settings_document: expected a JSON object.")
    return deepcopy(settings_document)


def normalize_viewer_settings(document, project_root):
    if not isinstance(document, dict):
        raise ViewerSettingsError("settings_document: expected a JSON object.")
    unknown = set(document) - set(DEFAULTS)
    if unknown:
        raise ViewerSettingsError("Unknown Viewer settings: " + ", ".join(sorted(unknown)))
    for key in REQUIRED:
        if key not in document:
            raise ViewerSettingsError(f"{key}: required.")
    result = deepcopy(DEFAULTS)
    for key, value in document.items():
        default = DEFAULTS[key]
        try:
            if default is None:
                if isinstance(value, bool):
                    raise ValueError("expected a number or null")
                value = None if value is None or str(value).strip().lower() in {"", "none"} else float(value)
            elif isinstance(default, bool):
                if not isinstance(value, bool):
                    text = str(value).lower()
                    if text not in {"true", "false", "1", "0"}:
                        raise ValueError("expected a boolean")
                    value = text in {"true", "1"}
            elif isinstance(default, (float, int)):
                if isinstance(value, bool):
                    raise ValueError("expected a number")
                number = float(value)
                if isinstance(default, int) and not number.is_integer():
                    raise ValueError("expected an integer")
                value = int(number) if isinstance(default, int) else number
            elif not isinstance(value, str):
                raise ValueError("expected text")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("expected a finite number")
            if key in PROFILE_ENUM_VALUES and value not in PROFILE_ENUM_VALUES[key]:
                raise ValueError("expected one of " + ", ".join(sorted(PROFILE_ENUM_VALUES[key])))
            if value is not None and key in PROFILE_RANGES:
                lo, hi = PROFILE_RANGES[key]
                if not lo <= value <= hi:
                    raise ValueError(f"expected {lo} to {hi}")
        except (TypeError, ValueError, OverflowError) as error:
            raise ViewerSettingsError(f"{key}: {error}") from error
        result[key] = value
    from matplotlib.colors import is_color_like
    for key in VISUAL_PROFILE_DEFAULTS:
        if key.endswith("COLOR") and not is_color_like(result[key]):
            raise ViewerSettingsError(f"{key}: invalid color.")
    if result["TARGET_CACHE_MODE"] != "existing":
        raise ViewerSettingsError("TARGET_CACHE_MODE: generate the cache before launching the Viewer.")
    if result["UMAP_MODE"]:
        if "UMAP_NEIGHBORS" not in document:
            raise ViewerSettingsError("UMAP_NEIGHBORS: required in UMAP mode.")
        if result["TOP_EDGE_PERCENT"] is not None or result["SIMILARITY_THRESHOLD"] is not None:
            raise ViewerSettingsError("UMAP_MODE: inactive edge filters must be null.")
    else:
        for key in ("SIMILARITY_THRESHOLD", "TOP_EDGE_PERCENT"):
            if key not in document:
                raise ViewerSettingsError(f"{key}: supply the active filter and explicitly set the inactive filter to null.")
        if result["TOP_EDGE_PERCENT"] is not None and result["SIMILARITY_THRESHOLD"] is not None:
            raise ViewerSettingsError("TOP_EDGE_PERCENT: SIMILARITY_THRESHOLD must be null when selecting top edges.")
    root = os.path.abspath(project_root)

    def resolve(value, base=None):
        value = os.path.expandvars(os.path.expanduser(value.strip()))
        for alias, key in ALIASES.items():
            if value == alias or value.startswith((alias + "/", alias + "\\")):
                value = os.path.join(result[key], value[len(alias):].lstrip("/\\"))
                break
        if "$input_file$" in value or "$cache_file$" in value or "$analysis_result$" in value:
            raise ViewerSettingsError(f"Invalid directory alias: {value}")
        return os.path.abspath(os.path.join(base or root, value))

    for key in ALIASES.values():
        result[key] = resolve(result[key])
    for key in (*DIRECTORY_PROFILE_DEFAULTS, "SAVED_CONFIG_DIR"):
        if key not in ALIASES.values():
            result[key] = resolve(result[key])
    for key, directory in FILES.items():
        value = result[key].strip()
        if not value:
            if key != "MSA_FILE":
                raise ViewerSettingsError(f"{key}: required nonempty path.")
            result[key] = ""
            continue
        # A cache-relative folder/version path is relative to SAVED_LAYOUT_DIR.
        base = result[directory] if key == "TARGET_CACHE_PATH" or not os.path.dirname(value) else root
        result[key] = resolve(value, base)
    return result


def validate_viewer_document(document, project_root):
    """Normalize and verify source/cache identity without changing configuration."""
    result = normalize_viewer_settings(document, project_root)
    for key in FILES:
        if result[key] and not os.path.isfile(result[key]):
            raise ViewerSettingsError(f"{key}: file does not exist: {result[key]}")
    import h5py
    import Cache_Manifest as manifest
    from utilities.FASTA_Sanitization import load_sanitized_fasta

    try:
        metadata = manifest.validate_network_schema(result["INPUT_HDF5"])
        if metadata.network_type == "alignment":
            for key in ("ALIGNMENT_SCORE", "NORM_MODE"):
                if key not in document:
                    raise ViewerSettingsError(f"{key}: required for alignment networks.")
            if result["ALIGNMENT_SCORE"] == "local" and result["NORM_MODE"] == "alignment_length":
                raise ViewerSettingsError("NORM_MODE: alignment_length is unavailable for local scores.")
        current = manifest.build_manifest_for_files(
            result["NODE_FASTA_FILE"], result["INPUT_HDF5"],
            alignment_score=result["ALIGNMENT_SCORE"], normalization=result["NORM_MODE"],
            umap_mode=result["UMAP_MODE"], umap_neighbors=result["UMAP_NEIGHBORS"],
            top_edge_percent=result["TOP_EDGE_PERCENT"], similarity_threshold=result["SIMILARITY_THRESHOLD"],
        )
        stored = manifest.read_manifest(os.path.dirname(result["TARGET_CACHE_PATH"]), current["compatibility"])
        selected, _, _ = load_sanitized_fasta(result["NODE_FASTA_FILE"], report=False)
        exact, identifiers = set(selected), {h.split()[0] for h in selected}
        with h5py.File(result["INPUT_HDF5"], "r") as network:
            headers = [h.decode() if isinstance(h, bytes) else str(h) for h in network["headers"][:]]
        headers = [h for h in headers if h in exact or h.split()[0] in identifiers]
        with h5py.File(result["TARGET_CACHE_PATH"], "r") as cache:
            manifest.validate_cache_hdf5(cache, headers, stored["manifest_id"])
        if result["MSA_FILE"]:
            if result["MSA_FILE"].lower().endswith(".h5"):
                with h5py.File(result["MSA_FILE"], "r") as msa:
                    msa_headers = [h.decode() if isinstance(h, bytes) else str(h) for h in msa["headers"][:]]
            else:
                with open(result["MSA_FILE"], encoding="utf-8") as msa:
                    msa_headers = [line[1:].strip() for line in msa if line.startswith(">")]
            reference = result["ALIGNMENT_REFERENCE"].strip().lower()
            if reference and not any(reference in h.lower() for h in msa_headers):
                raise ViewerSettingsError("ALIGNMENT_REFERENCE: not found in MSA_FILE.")
        elif result["ALIGNMENT_REFERENCE"]:
            raise ViewerSettingsError("ALIGNMENT_REFERENCE: requires MSA_FILE; use an empty reference when alignment is disabled.")
    except ViewerSettingsError:
        raise
    except Exception as error:
        raise ViewerSettingsError(f"Viewer input/cache validation: {error}") from error
    return result
