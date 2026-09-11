# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Consolidated Viewer document state, defaults, schemas, cache selection, and network preparation."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path

import numpy as np

import Cache_Manifest as cache_manifest

# =====================================================================
# 1. Aliases & Profile Defaults
# =====================================================================

INPUT_FILE_ALIAS = "$input_file$"
CACHE_FILE_ALIAS = "$cache_file$"
ANALYSIS_RESULT_ALIAS = "$analysis_result$"

INPUT_PROFILE_DEFAULTS = {
    "NODE_FASTA_FILE": "",
    "MSA_FILE": "",
    "INPUT_HDF5": "",
    "ALIGNMENT_SCORE": "global",
    "NORM_MODE": "alignment_length",
    "ALIGNMENT_REFERENCE": "",
    "ALIGNMENT_OFFSET": 0,
    "UMAP_MODE": False,
    "UMAP_NEIGHBORS": 15,
    "UMAP_MIN_DIST": 0.1,
    "SIMILARITY_THRESHOLD": None,
    "TOP_EDGE_PERCENT": None,
    "FILTER_MIN_OCCUPANCY": 10.0,
}

VISUAL_PROFILE_DEFAULTS = {
    "NODE_SIZE": 10,
    "EDGE_WIDTH": 1.0,
    "NODE_BOUNDARY_WIDTH": 0.5,
    "EDGE_ALPHA": 0.1,
    "TEXT_SIZE": 8,
    "TEXT_COLOR": "grey",
    "INITIAL_NODE_COLOR": "#4488ff",
    "HOVER_COLOR": "#ffaa00",
    "CONNECTED_NODE_COLOR": "#ff0000",
    "EDGE_COLOR": "#000000",
    "NODE_BOUNDARY_COLOR": "#000000",
    "LOW_RESOURCE_MODE": False,
}

PHYSICS_PROFILE_DEFAULTS = {
    "LAYOUT_DEVICE_SELECTION": "auto",
    "SPRING_K": 5.0,
    "COULOMB_K": 10.0,
    "COULOMB_CUTOFF": 30.0,
    "DAMPING": 0.9,
    "DT": 0.005,
    "MAX_STEPS": 10000,
    "RMSD_THRESHOLD": 0.005,
    "PERCENTAGE_DROP_THRESHOLD": 0.1,
    "RMSD_WINDOW": 50,
    "ENABLE_PROGRESSIVE_SIMULATION": False,
    "PACKING_GEOMETRY": "Square",
    "PACKING_GRID_SIZE": 20.0,
}

DIRECTORY_PROFILE_DEFAULTS = {
    "INPUT_FILE_DIR": "Input_Files",
    "CACHE_FILE_DIR": "Cache_Files",
    "ANALYSIS_RESULT_DIR": "Analysis_Results",
    "FASTA_DIR": os.path.join(INPUT_FILE_ALIAS, "Sequence_Sets"),
    "MSA_DIR": os.path.join(INPUT_FILE_ALIAS, "Multiple_Alignments"),
    "HDF5_DIR": os.path.join(INPUT_FILE_ALIAS, "Networks_EValues"),
    "METADATA_DIR": os.path.join(INPUT_FILE_ALIAS, "Meta_Data"),
    "HEADER_LIST_DIR": os.path.join(INPUT_FILE_ALIAS, "Header_Lists"),
    "SAVED_LAYOUT_DIR": os.path.join(CACHE_FILE_ALIAS, "Saved_Layouts"),
    "SETTING_EXPORT_DIR": os.path.join(CACHE_FILE_ALIAS, "Exported_Settings"),
}

LEGACY_DEFAULT_DIRECTORY_PATHS = {
    "FASTA_DIR": os.path.join("Input_Files", "Sequence_Sets"),
    "MSA_DIR": os.path.join("Input_Files", "Multiple_Alignments"),
    "HDF5_DIR": os.path.join("Input_Files", "Networks_EValues"),
    "METADATA_DIR": os.path.join("Input_Files", "Meta_Data"),
    "HEADER_LIST_DIR": os.path.join("Input_Files", "Header_Lists"),
    "SAVED_LAYOUT_DIR": os.path.join("Cache_Files", "Saved_Layouts"),
    "SETTING_EXPORT_DIR": os.path.join("Cache_Files", "Exported_Settings"),
}

PROFILE_ENUM_VALUES = {
    "ALIGNMENT_SCORE": {"global", "local"},
    "NORM_MODE": {
        "alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"
    },
    "PACKING_GEOMETRY": {"Square", "Circle"},
}

PROFILE_RANGES = {
    "ALIGNMENT_OFFSET": (-1000000, 1000000),
    "UMAP_NEIGHBORS": (2, 500),
    "UMAP_MIN_DIST": (0.0, 1.0),
    "TOP_EDGE_PERCENT": (0.0, 100.0),
    "FILTER_MIN_OCCUPANCY": (0.0, 100.0),
    "NODE_SIZE": (1, 20),
    "EDGE_WIDTH": (0.1, 3.0),
    "NODE_BOUNDARY_WIDTH": (0.0, 2.0),
    "EDGE_ALPHA": (0.0, 1.0),
    "TEXT_SIZE": (1, 24),
    "SPRING_K": (1.0, 20.0),
    "COULOMB_K": (1.0, 30.0),
    "COULOMB_CUTOFF": (1.0, 100.0),
    "DAMPING": (0.1, 2.0),
    "RMSD_WINDOW": (10, 1000),
    "PACKING_GRID_SIZE": (1.0, 200.0),
}

# =====================================================================
# 2. Execution Settings Document Schema & Codec
# =====================================================================

LAYOUT_SECTIONS = {
    "inputs": ("NODE_FASTA_FILE", "INPUT_HDF5"),
    "network": ("ALIGNMENT_SCORE", "NORM_MODE", "SIMILARITY_THRESHOLD", "TOP_EDGE_PERCENT"),
    "layout": ("UMAP_MODE", "UMAP_NEIGHBORS", "UMAP_MIN_DIST"),
    "simulation": ("LAYOUT_DEVICE_SELECTION", "DT", "MAX_STEPS", "RMSD_THRESHOLD", "PERCENTAGE_DROP_THRESHOLD", "RMSD_WINDOW", "ENABLE_PROGRESSIVE_SIMULATION"),
    "physics": ("SPRING_K", "COULOMB_K", "COULOMB_CUTOFF", "DAMPING", "MAX_FORCE_LIMIT", "MAX_TOTAL_REPULSION_FORCE"),
    "packing": ("PACKING_GEOMETRY", "PACKING_GRID_SIZE", "BOX_SCALE", "PACKING_PADDING"),
    "output": ("SAVED_LAYOUT_DIR", "TARGET_CACHE_PATH", "CACHE_FILENAME", "CACHE_NAME_MODE"),
}

VIEWER_SECTIONS = {
    "inputs": ("TARGET_CACHE_PATH", "NODE_FASTA_FILE", "INPUT_HDF5"),
    "alignment": ("MSA_FILE", "ALIGNMENT_REFERENCE", "FILTER_MIN_OCCUPANCY", "ALIGNMENT_OFFSET"),
    "visualization": tuple(VISUAL_PROFILE_DEFAULTS),
    "directories": tuple(DIRECTORY_PROFILE_DEFAULTS),
}


def sections(kind):
    return {"layout": LAYOUT_SECTIONS, "viewer": VIEWER_SECTIONS}[kind]


def encode_document(kind, values):
    return {
        "schema_version": 2,
        "kind": kind,
        **{
            section: {key: deepcopy(values[key]) for key in keys}
            for section, keys in sections(kind).items()
        },
    }


def decode_document(document, kind, *, partial=False):
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document.get("schema_version") != 2 or document.get("kind") != kind:
        raise ValueError(f"Expected schema_version 2, kind '{kind}'. Re-export settings through the GUI, CLI, or MCP export action; legacy execution JSON is unsupported.")
    mapping = sections(kind)
    unknown = set(document) - {"schema_version", "kind", *mapping}
    if unknown:
        raise ValueError("Unknown document sections: " + ", ".join(sorted(unknown)) + ". Re-export settings.")
    result = {}
    for section, keys in mapping.items():
        values = document.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section}: expected an object. Re-export settings.")
        unknown = set(values) - set(keys)
        missing = set(keys) - set(values)
        if unknown:
            raise ValueError(f"{section}: unknown or misplaced fields: " + ", ".join(sorted(unknown)) + ". Re-export settings.")
        if missing and not partial:
            raise ValueError(f"{section}: missing fields: " + ", ".join(sorted(missing)) + ". Re-export settings.")
        result.update(deepcopy(values))
    return result


# =====================================================================
# 3. Viewer Settings Normalization, Validation & Provenance
# =====================================================================

class ViewerSettingsError(ValueError):
    pass


DEFAULTS = {
    **DIRECTORY_PROFILE_DEFAULTS, **INPUT_PROFILE_DEFAULTS,
    **VISUAL_PROFILE_DEFAULTS, **PHYSICS_PROFILE_DEFAULTS,
    "TARGET_CACHE_PATH": "", "TARGET_CACHE_MODE": "existing",
    "CACHE_FILENAME": "",
    "SAVED_CONFIG_DIR": "$cache_file$/Saved_Config", "BOX_SCALE": 2.0,
    "PACKING_PADDING": 10.0, "MAX_FORCE_LIMIT": 20.0,
    "MAX_TOTAL_REPULSION_FORCE": 0.0,
}
REQUIRED = ("TARGET_CACHE_PATH", "NODE_FASTA_FILE", "INPUT_HDF5", "MSA_FILE", "UMAP_MODE")
ALIASES = {
    "$input_file$": "INPUT_FILE_DIR",
    "$cache_file$": "CACHE_FILE_DIR",
    "$analysis_result$": "ANALYSIS_RESULT_DIR",
}
FILES = {
    "NODE_FASTA_FILE": "FASTA_DIR",
    "INPUT_HDF5": "HDF5_DIR",
    "MSA_FILE": "MSA_DIR",
    "TARGET_CACHE_PATH": "SAVED_LAYOUT_DIR",
}


def _flat_viewer_settings_schema():
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
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(REQUIRED),
        "allOf": [
            {
                "if": {"properties": {"UMAP_MODE": {"const": True}}},
                "then": {"required": ["UMAP_NEIGHBORS"]},
                "else": {"required": ["SIMILARITY_THRESHOLD", "TOP_EDGE_PERCENT"]},
            },
        ],
        "description": "Complete Viewer settings. Alignment networks also require ALIGNMENT_SCORE and NORM_MODE. MSA_FILE='' disables alignment. Scientific settings must match the cache manifest. Relative file selectors use their configured directory; paths with directories use project root.",
    }


def get_viewer_settings_schema():
    flat = _flat_viewer_settings_schema()["properties"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "kind", *VIEWER_SECTIONS],
        "properties": {
            "schema_version": {"const": 2},
            "kind": {"const": "viewer"},
            **{
                section: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(keys),
                    "properties": {key: flat[key] for key in keys},
                }
                for section, keys in VIEWER_SECTIONS.items()
            },
        },
        "description": "Export first. Viewer preferences only; generation settings are resolved from verified cache provenance. Legacy JSON is rejected.",
    }


def resolve_cache_settings(path):
    from utilities.Cache_Metadata import read_cache_metadata
    metadata = read_cache_metadata(path)
    if metadata["status"] != "complete":
        raise ViewerSettingsError("Cache provenance: " + "; ".join(metadata["diagnostics"]))
    try:
        compatibility = metadata["folder_manifest"]["compatibility"]
        parameters = metadata["generation_parameters"]
        mode = compatibility["layout_mode"]
        if mode not in {"physics", "umap"}:
            raise ValueError("invalid layout_mode")
        values = {key: parameters[key] for key in ("UMAP_MODE", "UMAP_NEIGHBORS", "UMAP_MIN_DIST", "BOX_SCALE")}
        if type(values["UMAP_MODE"]) is not bool or values["UMAP_MODE"] != (mode == "umap"):
            raise ValueError("conflicting UMAP_MODE")
        if type(values["UMAP_NEIGHBORS"]) is not int or not 2 <= values["UMAP_NEIGHBORS"] <= 500:
            raise ValueError("invalid UMAP_NEIGHBORS")
        for key in ("UMAP_MIN_DIST", "BOX_SCALE"):
            if isinstance(values[key], bool) or not isinstance(values[key], (int, float)) or not math.isfinite(values[key]):
                raise ValueError(f"invalid {key}")
        if not 0 <= values["UMAP_MIN_DIST"] <= 1 or values["BOX_SCALE"] <= 0:
            raise ValueError("invalid UMAP_MIN_DIST or BOX_SCALE")
        values.update(ALIGNMENT_SCORE=compatibility["alignment_score"], NORM_MODE=compatibility["normalization"],
                      SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=None)
        edge = compatibility["edge_filter"]
        expected = "umap_neighbors" if mode == "umap" else edge["mode"]
        if expected == "umap_neighbors" and mode == "umap":
            if edge["mode"] != expected or edge["value"] != values["UMAP_NEIGHBORS"]:
                raise ValueError("conflicting UMAP neighbor filter")
        elif expected in {"similarity_threshold", "top_edge_percent"} and mode == "physics":
            value = edge["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("invalid edge filter")
            values[expected.upper()] = value
        else:
            raise ValueError("invalid edge filter mode")
        threshold = parameters["SIMILARITY_THRESHOLD"]
        if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)):
            raise ValueError("invalid generation similarity threshold")
        if threshold != values["SIMILARITY_THRESHOLD"]:
            raise ValueError("conflicting similarity threshold")
        if compatibility["network_type"] == "blast":
            if values["ALIGNMENT_SCORE"] is not None or values["NORM_MODE"] is not None:
                raise ValueError("BLAST score settings must be null")
            values.update(ALIGNMENT_SCORE="global", NORM_MODE="alignment_length")
        elif compatibility["network_type"] != "alignment":
            raise ValueError("invalid network_type")
        return values
    except (KeyError, TypeError, ValueError) as error:
        raise ViewerSettingsError(f"Cache provenance: {error}") from error


def normalize_viewer_paths(values, project_root, *, require_cache=True):
    """Resolve Config paths without validating unrelated display/physics controls."""
    result = deepcopy(values)
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
            if key != "MSA_FILE" and (key != "TARGET_CACHE_PATH" or require_cache):
                raise ViewerSettingsError(f"{key}: required nonempty path.")
            result[key] = ""
            continue
        base = result[directory] if key == "TARGET_CACHE_PATH" or not os.path.dirname(value) else root
        result[key] = resolve(value, base)
    filename = os.path.basename(result["TARGET_CACHE_PATH"])
    if result["CACHE_FILENAME"] and result["CACHE_FILENAME"] != filename:
        raise ViewerSettingsError("CACHE_FILENAME must match TARGET_CACHE_PATH.")
    result["CACHE_FILENAME"] = filename
    return result


def normalize_viewer_settings(document, project_root, *, require_cache=True):
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
    return normalize_viewer_paths(result, project_root, require_cache=require_cache)


def _validate_flat_viewer_document(document, project_root):
    """Normalize and verify source/cache identity without changing configuration."""
    result = normalize_viewer_settings(document, project_root)
    for key in FILES:
        if result[key] and not os.path.isfile(result[key]):
            raise ViewerSettingsError(f"{key}: file does not exist: {result[key]}")
    import h5py
    import Cache_Manifest as manifest
    from utilities.Sequence_Utils import load_sanitized_fasta

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
            # Preserve header readability checks. Alignment_Manager resolves the
            # requested reference and warns/falls back to occupancy if absent.
            if result["MSA_FILE"].lower().endswith(".h5"):
                with h5py.File(result["MSA_FILE"], "r") as msa:
                    _msa_headers = [h.decode() if isinstance(h, bytes) else str(h) for h in msa["headers"][:]]
            else:
                with open(result["MSA_FILE"], encoding="utf-8") as msa:
                    _msa_headers = [line[1:].strip() for line in msa if line.startswith(">")]
        elif result["ALIGNMENT_REFERENCE"]:
            raise ViewerSettingsError("ALIGNMENT_REFERENCE: requires MSA_FILE; use an empty reference when alignment is disabled.")
    except ViewerSettingsError:
        raise
    except Exception as error:
        raise ViewerSettingsError(f"Viewer input/cache validation: {error}") from error
    return result


def resolve_viewer_document(document, project_root):
    try:
        values = decode_document(document, "viewer")
    except ValueError as error:
        raise ViewerSettingsError(str(error)) from error
    values.update(UMAP_MODE=False, SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=None)
    normalized = normalize_viewer_settings(values, project_root)
    normalized.update(resolve_cache_settings(normalized["TARGET_CACHE_PATH"]))
    return _validate_flat_viewer_document(normalized, project_root)


def validate_viewer_document(document, project_root):
    return encode_document("viewer", resolve_viewer_document(document, project_root))


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


# =====================================================================
# 4. Cache Resolution & Selection
# =====================================================================

def _configured_reference_text(settings):
    value = getattr(settings, "ALIGNMENT_REFERENCE", None)
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "none" else text


def _resolve_reference_header(settings):
    reference_text = _configured_reference_text(settings)
    msa_file = getattr(settings, "MSA_FILE", None)
    if not reference_text or not msa_file or not os.path.exists(msa_file):
        return None

    try:
        reference_lower = reference_text.lower()
        if os.path.splitext(os.fspath(msa_file))[1].lower() == ".h5":
            import h5py

            with h5py.File(msa_file, "r") as handle:
                if "headers" not in handle:
                    return None
                for raw_header in handle["headers"][:]:
                    header = (
                        raw_header.decode("utf-8")
                        if isinstance(raw_header, bytes)
                        else raw_header
                    )
                    if reference_lower in header.lower():
                        return header
        else:
            with open(msa_file, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith(">") and reference_lower in line.lower():
                        return line.strip()[1:]
    except Exception as error:
        print(f"Utils Warning: Could not resolve reference header: {error}")
    return None


def resolve_selected_cache(settings):
    """Return ``(cache_path, resolved_reference_header)`` for ``settings``."""
    resolved_reference = _resolve_reference_header(settings)
    saved_layout_dir = getattr(
        settings,
        "SAVED_LAYOUT_DIR",
        os.path.join("Cache_Files", "Saved_Layouts"),
    )

    explicit_path = getattr(settings, "TARGET_CACHE_PATH", None)
    if explicit_path:
        if os.path.isabs(explicit_path):
            return os.path.abspath(explicit_path), resolved_reference
        return (
            cache_manifest.resolve_relative_cache_path(
                saved_layout_dir, explicit_path
            ),
            resolved_reference,
        )

    fasta_file = getattr(settings, "NODE_FASTA_FILE", None) or getattr(
        settings, "SEQUENCES_FILE", ""
    )
    network_file = getattr(settings, "INPUT_HDF5", "")
    network_type = cache_manifest.validate_network_schema(network_file).network_type
    settings.INPUT_IS_EVALUE = network_type == "blast"
    canonical_name = cache_manifest.build_canonical_cache_name(
        fasta_file,
        network_file,
        network_type,
        alignment_score=getattr(settings, "ALIGNMENT_SCORE", None),
        normalization=getattr(settings, "NORM_MODE", None),
        umap_mode=getattr(settings, "UMAP_MODE", False),
        umap_neighbors=getattr(settings, "UMAP_NEIGHBORS", 15),
        top_edge_percent=getattr(settings, "TOP_EDGE_PERCENT", None),
        similarity_threshold=getattr(settings, "SIMILARITY_THRESHOLD", None),
    )
    target_folder = os.path.join(saved_layout_dir, canonical_name)

    selected_cache = getattr(settings, "TARGET_CACHE_FILE", None)
    if (
        isinstance(selected_cache, str)
        and selected_cache.strip()
        and selected_cache != "None"
    ):
        cache_manifest.validate_cache_filename(selected_cache)
        return os.path.join(target_folder, selected_cache), resolved_reference

    return os.path.join(target_folder, "version_00.h5"), resolved_reference


# =====================================================================
# 5. Network Preparation (Filtering, Thresholding, UMAP)
# =====================================================================

def _normalize_score(raw_score, align_len, len_i, len_j, mode):
    if mode == "alignment_length":
        denominator = align_len
    elif mode == "shorter_sequence":
        denominator = np.minimum(len_i, len_j)
    elif mode == "longer_sequence":
        denominator = np.maximum(len_i, len_j)
    elif mode == "average_sequence":
        denominator = (len_i + len_j) / 2.0
    else:
        denominator = align_len

    return np.where(denominator > 0, raw_score / denominator, 0.0)


def prepare_network(data, *, settings, selected_fasta_headers=None):
    """Return filtered ``(headers, edges, scores)`` for an open network file."""
    metadata = cache_manifest.validate_network_schema(data)
    settings.INPUT_IS_EVALUE = metadata.network_type == "blast"

    raw_headers = data["headers"][:]
    headers = [
        header.decode("utf-8") if isinstance(header, bytes) else header
        for header in raw_headers
    ]
    total_nodes = len(headers)

    sources = data["i"][:]
    targets = data["j"][:]
    if settings.INPUT_IS_EVALUE:
        scores = data["score"][:]
    else:
        sequence_lengths = data["seq_lens"][:]
        if settings.ALIGNMENT_SCORE == "global":
            alignment_scores = data["g_score"][:]
            alignment_lengths = data["g_len"][:]
        else:
            alignment_scores = data["l_score"][:]
            alignment_lengths = data["l_len"][:]

    print(f"Raw Data: {total_nodes} sequences.")
    if not settings.INPUT_IS_EVALUE:
        print(
            f"Metric: {settings.ALIGNMENT_SCORE.upper()} Alignment with "
            f"{settings.NORM_MODE} Normalization"
        )

    fasta_path = getattr(settings, "NODE_FASTA_FILE", "")
    kept_indices = []
    if selected_fasta_headers is not None or os.path.exists(fasta_path):
        clean_fasta_path = os.path.normpath(fasta_path)
        print(f"Scanning FASTA file for node filter: {clean_fasta_path}")
        fasta_ids = set()
        fasta_headers = set()
        try:
            if selected_fasta_headers is None:
                from utilities.Sequence_Utils import load_sanitized_fasta
                selected_fasta_headers, _, _ = load_sanitized_fasta(fasta_path)

            for header in selected_fasta_headers:
                fasta_headers.add(header)
                header_parts = header.split()
                if header_parts:
                    fasta_ids.add(header_parts[0])

            network_headers = set(headers)
            network_ids = {header.split()[0] for header in headers}
            missing_nodes = [
                identifier
                for identifier in fasta_ids
                if identifier not in network_ids and identifier not in network_headers
            ]
            if missing_nodes:
                print(
                    "CRITICAL WARNING: The passed FASTA file is NOT a strict "
                    f"subset of the network file. {len(missing_nodes)} FASTA "
                    "sequences are missing from the network."
                )

            for index, header in enumerate(headers):
                record_id = header.split()[0]
                if header in fasta_headers or record_id in fasta_ids:
                    kept_indices.append(index)

            kept_indices = np.asarray(kept_indices, dtype=np.int64)
            print(
                f"Filtered {total_nodes} down to {len(kept_indices)} valid "
                "FASTA subsets."
            )
        except Exception as error:
            print(f"Error reading FASTA filter: {error}. Retaining all sequences.")
            kept_indices = np.arange(total_nodes)
    else:
        print(
            f"No FASTA file found at {fasta_path}. Retaining all "
            f"{total_nodes} sequences."
        )
        kept_indices = np.arange(total_nodes)

    kept_mask = np.zeros(total_nodes, dtype=bool)
    kept_mask[kept_indices] = True
    filtered_headers = [headers[index] for index in kept_indices]

    index_map = np.zeros(total_nodes, dtype=np.int32)
    index_map[kept_indices] = np.arange(len(kept_indices))

    valid_edges_mask = kept_mask[sources] & kept_mask[targets]
    valid_sources = sources[valid_edges_mask]
    valid_targets = targets[valid_edges_mask]
    if settings.INPUT_IS_EVALUE:
        valid_scores = scores[valid_edges_mask]
    else:
        valid_raw_scores = alignment_scores[valid_edges_mask]
        valid_alignment_lengths = alignment_lengths[valid_edges_mask]
        valid_scores = _normalize_score(
            valid_raw_scores,
            valid_alignment_lengths,
            sequence_lengths[valid_sources],
            sequence_lengths[valid_targets],
            settings.NORM_MODE,
        )

    top_percent = getattr(settings, "TOP_EDGE_PERCENT", None)
    if top_percent is not None and not getattr(settings, "UMAP_MODE", False):
        active_nodes = len(kept_indices)
        theoretical_max_edges = (active_nodes * (active_nodes - 1)) / 2.0
        edge_count = int(theoretical_max_edges * (top_percent / 100.0))
        if len(valid_scores) == 0:
            calculated_cutoff = 0.0
        else:
            edge_count = max(1, min(edge_count, len(valid_scores)))
            calculated_cutoff = float(np.sort(valid_scores)[::-1][edge_count - 1])

        mode_label = "E-Value" if settings.INPUT_IS_EVALUE else "Similarity"
        print(
            f"Top {top_percent}% Edges Requested (based on max possible "
            f"{int(theoretical_max_edges)} edges)."
        )
        print(f"Calculated {mode_label} Cutoff: {calculated_cutoff:.5f}")
        settings.SIMILARITY_THRESHOLD = calculated_cutoff

    if getattr(settings, "UMAP_MODE", False):
        print(
            "UMAP Mode enabled: Bypassing global threshold. Filtering top k "
            "edges per node..."
        )
        keep_limit = int(getattr(settings, "UMAP_NEIGHBORS", 15))
        import pandas as pd

        frame = pd.DataFrame(
            {
                "u": valid_sources,
                "v": valid_targets,
                "score": valid_scores,
                "idx": np.arange(len(valid_scores)),
            }
        )
        sorted_frame = frame.sort_values("score", ascending=False)
        top_sources = sorted_frame.groupby("u").head(keep_limit)["idx"]
        top_targets = sorted_frame.groupby("v").head(keep_limit)["idx"]
        kept_edge_indices = pd.concat([top_sources, top_targets]).unique()
        threshold_mask = np.zeros(len(valid_scores), dtype=bool)
        threshold_mask[kept_edge_indices] = True
        print(
            f"Kept {len(kept_edge_indices)} edges for UMAP topology "
            f"(max {keep_limit} per node direction)."
        )
    else:
        threshold_mask = valid_scores >= settings.SIMILARITY_THRESHOLD

    final_sources = index_map[valid_sources[threshold_mask]]
    final_targets = index_map[valid_targets[threshold_mask]]
    edges = np.column_stack((final_sources, final_targets)).astype(np.int32)
    edge_scores = valid_scores[threshold_mask]
    return filtered_headers, edges, edge_scores


__all__ = [
    "INPUT_FILE_ALIAS",
    "CACHE_FILE_ALIAS",
    "ANALYSIS_RESULT_ALIAS",
    "INPUT_PROFILE_DEFAULTS",
    "VISUAL_PROFILE_DEFAULTS",
    "PHYSICS_PROFILE_DEFAULTS",
    "DIRECTORY_PROFILE_DEFAULTS",
    "LEGACY_DEFAULT_DIRECTORY_PATHS",
    "PROFILE_ENUM_VALUES",
    "PROFILE_RANGES",
    "LAYOUT_SECTIONS",
    "VIEWER_SECTIONS",
    "sections",
    "encode_document",
    "decode_document",
    "ViewerSettingsError",
    "DEFAULTS",
    "REQUIRED",
    "ALIASES",
    "FILES",
    "get_viewer_settings_schema",
    "resolve_cache_settings",
    "resolve_viewer_document",
    "validate_viewer_document",
    "read_viewer_settings",
    "normalize_viewer_settings",
    "normalize_viewer_paths",
    "resolve_selected_cache",
    "prepare_network",
]
