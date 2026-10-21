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
EMBEDDING_ALIAS = "$embedding$"

INPUT_PROFILE_DEFAULTS = {
    "NODE_FASTA_FILE": "",
    "MSA_FILE": "",
    "EMBED_FILE": "",
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
}

# The Visual Effects tab's Low Resource Mode: a switch per category of work
# the Viewer can be spared, all off by default. The Viewer asks each through
# its helper below, so a category can come to cover more without new reads.
LOW_RESOURCE_DEFAULTS = {
    # Basic Animations: the wheel zoom without its glide, the sidebar without its slide.
    "LOW_RESOURCE_ANIMATIONS": False,
    # Node and Edge Drawing: the threshold slider redraws the edges when let
    # go; a right-drag of nodes leaves their edges out, and a selection box
    # colours its nodes and their edges, only once the button is released.
    "LOW_RESOURCE_DRAWING": False,
    # Webserver Communication: the Meta Data sheet hears of a node drag's
    # selection when the drag ends.
    "LOW_RESOURCE_WEBSERVER": False,
}
VISUAL_PROFILE_DEFAULTS.update({
    # Fewer Edges While Moving: every k-th edge while the view moves
    # (desktop.Viewer_Motion). The first category; the old single switch
    # never stood for it.
    "FEWER_EDGES_WHILE_MOVING": False,
    **LOW_RESOURCE_DEFAULTS,
    # The frame rate over the canvas's top right (desktop.Viewer_Fps).
    "SHOW_FPS": False,
})


def fewer_edges_while_moving(configuration):
    """Fewer Edges While Moving: whether the Viewer may thin the edges
    while the view moves (SSN_EDGE_MOTION_LOD=0 still turns it off)."""
    return bool(getattr(configuration, "FEWER_EDGES_WHILE_MOVING", False))


def low_resource_animations(configuration):
    """Basic Animations: whether the wheel zoom and the sidebar skip their animations."""
    return bool(getattr(configuration, "LOW_RESOURCE_ANIMATIONS", False))


def low_resource_drawing(configuration):
    """Node and Edge Drawing: whether the Viewer redraws the edges and the
    selection's colouring only when a slider drag, a node drag or a
    selection box ends."""
    return bool(getattr(configuration, "LOW_RESOURCE_DRAWING", False))


def low_resource_webserver(configuration):
    """Webserver Communication: whether the web pages hear of a node drag's
    selection only when the drag ends."""
    return bool(getattr(configuration, "LOW_RESOURCE_WEBSERVER", False))


# The one switch the categories replaced, which turned Basic Animations, Node
# and Edge Drawing and Webserver Communication on or off together. Settings
# files, saved profiles and settings documents written before may hold it.
LEGACY_LOW_RESOURCE_KEY = "LOW_RESOURCE_MODE"


def migrate_low_resource_mode(values):
    """values with LOW_RESOURCE_MODE taken out, as the categories it stood for.

    When values holds none of LOW_RESOURCE_DEFAULTS, each takes the old key's
    value; Fewer Edges While Moving keeps its own (off unless set). A
    category already there wins, and the old key is dropped either way. A
    dict without the old key is returned as it is.
    """
    if not isinstance(values, dict) or LEGACY_LOW_RESOURCE_KEY not in values:
        return values
    result = dict(values)
    legacy = result.pop(LEGACY_LOW_RESOURCE_KEY)
    if not any(key in result for key in LOW_RESOURCE_DEFAULTS):
        for key in LOW_RESOURCE_DEFAULTS:
            result[key] = legacy
    return result

PHYSICS_PROFILE_DEFAULTS = {
    "LAYOUT_DEVICE_SELECTION": "auto",
    "SPRING_K": 5.0,
    "COULOMB_K": 10.0,
    "COULOMB_CUTOFF": 30.0,
    "DAMPING": 0.9,
    "DT": 0.005,
    "AUTO_DT": False,
    "MAX_STEPS": 10000,
    "RMSD_THRESHOLD": 0.005,
    "PERCENTAGE_DROP_THRESHOLD": 0.1,
    "RMSD_WINDOW": 50,
    "ENABLE_PROGRESSIVE_SIMULATION": False,
    "PACKING_GEOMETRY": "Square",
    "PACKING_GRID_SIZE": 10.0,
}

DIRECTORY_PROFILE_DEFAULTS = {
    "INPUT_FILE_DIR": "Input_Files",
    "CACHE_FILE_DIR": "Cache_Files",
    "ANALYSIS_RESULT_DIR": "Analysis_Results",
    # A root of its own, beside Input_Files and not in it: the folder the Tools
    # write embeddings to (tool_helpers.Tool_Pipeline.DEFAULT_DIRECTORY_PATHS).
    "EMBED_DIR": "Embeddings",
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
    "layout": ("UMAP_MODE", "UMAP_NEIGHBORS", "UMAP_MIN_DIST", "LAYOUT_DIMENSIONS", "LAYOUT_SEED"),
    "simulation": ("LAYOUT_DEVICE_SELECTION", "DT", "AUTO_DT", "MAX_STEPS", "RMSD_THRESHOLD", "PERCENTAGE_DROP_THRESHOLD", "RMSD_WINDOW", "ENABLE_PROGRESSIVE_SIMULATION"),
    "physics": ("SPRING_K", "COULOMB_K", "COULOMB_CUTOFF", "DAMPING", "MAX_FORCE_LIMIT", "MAX_TOTAL_REPULSION_FORCE"),
    "packing": ("PACKING_GEOMETRY", "PACKING_GRID_SIZE", "BOX_SCALE", "PACKING_PADDING"),
    "output": ("SAVED_LAYOUT_DIR", "TARGET_CACHE_PATH", "CACHE_FILENAME", "CACHE_NAME_MODE"),
}

VIEWER_SECTIONS = {
    "inputs": ("TARGET_CACHE_PATH", "NODE_FASTA_FILE", "INPUT_HDF5"),
    "alignment": ("MSA_FILE", "ALIGNMENT_REFERENCE", "FILTER_MIN_OCCUPANCY", "ALIGNMENT_OFFSET"),
    "embedding": ("EMBED_FILE",),
    "visualization": tuple(VISUAL_PROFILE_DEFAULTS),
    "directories": tuple(DIRECTORY_PROFILE_DEFAULTS),
}


# Keys a document may omit. Absent ones decode to the default below, so
# documents exported before the key existed remain valid.
OPTIONAL_DOCUMENT_KEYS = {
    "layout": {"LAYOUT_DIMENSIONS": 2, "LAYOUT_SEED": 42, "AUTO_DT": False},
    "viewer": {
        "EMBED_FILE": INPUT_PROFILE_DEFAULTS["EMBED_FILE"],
        "EMBED_DIR": DIRECTORY_PROFILE_DEFAULTS["EMBED_DIR"],
        # The Low Resource Mode categories and Show FPS came after
        # LOW_RESOURCE_MODE, which a document may hold instead
        # (migrate_low_resource_mode).
        **{key: VISUAL_PROFILE_DEFAULTS[key] for key in (
            "FEWER_EDGES_WHILE_MOVING", *LOW_RESOURCE_DEFAULTS, "SHOW_FPS")},
    },
}


def sections(kind):
    return {"layout": LAYOUT_SECTIONS, "viewer": VIEWER_SECTIONS}[kind]


def optional_keys(kind):
    return OPTIONAL_DOCUMENT_KEYS[kind]


def encode_document(kind, values):
    optional = optional_keys(kind)

    def value_for(key):
        if key in values:
            return deepcopy(values[key])
        if key in optional:
            return deepcopy(optional[key])
        raise KeyError(key)

    return {
        "schema_version": 2,
        "kind": kind,
        **{
            section: {key: value_for(key) for key in keys}
            for section, keys in sections(kind).items()
        },
    }


def decode_document(document, kind, *, partial=False):
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document.get("schema_version") != 2 or document.get("kind") != kind:
        raise ValueError(f"Expected schema_version 2, kind '{kind}'. Re-export settings through the GUI, CLI, or MCP export action; legacy execution JSON is unsupported.")
    mapping = sections(kind)
    optional = optional_keys(kind)
    unknown = set(document) - {"schema_version", "kind", *mapping}
    if unknown:
        raise ValueError("Unknown document sections: " + ", ".join(sorted(unknown)) + ". Re-export settings.")
    result = {}
    for section, keys in mapping.items():
        values = document.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section}: expected an object. Re-export settings.")
        if kind == "viewer" and section == "visualization":
            values = migrate_low_resource_mode(values)
        unknown = set(values) - set(keys)
        missing = set(keys) - set(values) - set(optional)
        if unknown:
            raise ValueError(f"{section}: unknown or misplaced fields: " + ", ".join(sorted(unknown)) + ". Re-export settings.")
        if missing and not partial:
            raise ValueError(f"{section}: missing fields: " + ", ".join(sorted(missing)) + ". Re-export settings.")
        result.update(deepcopy(values))
    for key, default in optional.items():
        result.setdefault(key, deepcopy(default))
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
    "$embedding$": "EMBED_DIR",
}
FILES = {
    "NODE_FASTA_FILE": "FASTA_DIR",
    "INPUT_HDF5": "HDF5_DIR",
    "MSA_FILE": "MSA_DIR",
    "EMBED_FILE": "EMBED_DIR",
    "TARGET_CACHE_PATH": "SAVED_LAYOUT_DIR",
}
# Optional inputs: an empty path selects none.
OPTIONAL_FILES = frozenset({"MSA_FILE", "EMBED_FILE"})


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
        "description": "Complete Viewer settings. Alignment networks also require ALIGNMENT_SCORE and NORM_MODE. MSA_FILE='' disables alignment, and EMBED_FILE='' (the default) the embedding. Scientific settings must match the cache manifest. Relative file selectors use their configured directory; paths with directories use project root.",
    }


# Fields older documents may hold, which decode_document maps to today's.
_LEGACY_VISUALIZATION_PROPERTIES = {
    LEGACY_LOW_RESOURCE_KEY: {
        "type": "boolean",
        "deprecated": True,
        "description": "Older documents' single Low Resource Mode switch. When the section has "
                       "none of LOW_RESOURCE_ANIMATIONS, LOW_RESOURCE_DRAWING and "
                       "LOW_RESOURCE_WEBSERVER, each takes this value; otherwise it is ignored. "
                       "Exports write the categories instead.",
    },
}


def get_viewer_settings_schema():
    flat = _flat_viewer_settings_schema()["properties"]
    optional = optional_keys("viewer")
    # A section whose keys are all optional may be left out, as in documents
    # exported before it existed.
    required_sections = [
        section for section, keys in VIEWER_SECTIONS.items()
        if any(key not in optional for key in keys)
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "kind", *required_sections],
        "properties": {
            "schema_version": {"const": 2},
            "kind": {"const": "viewer"},
            **{
                section: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [key for key in keys if key not in optional],
                    "properties": {
                        **{key: flat[key] for key in keys},
                        **(_LEGACY_VISUALIZATION_PROPERTIES if section == "visualization" else {}),
                    },
                }
                for section, keys in VIEWER_SECTIONS.items()
            },
        },
        "description": "Export first. Viewer preferences only; generation settings are resolved from verified cache provenance. Legacy JSON is rejected. alignment.MSA_FILE='' disables alignment. The embedding section and EMBED_DIR may be omitted: embedding.EMBED_FILE='' selects no embedding, and a selected file must be a complete protein language model embedding database (.h5) as the pipeline writes it; it may cover only some nodes.",
    }


def resolve_cache_settings(path, *, allow_3d=False):
    """Return the analysis settings the cache at ``path`` was generated with.

    The desktop Viewer opens 2D caches only. The optional VR viewer (opt_vr)
    passes ``allow_3d=True`` to read the 3D caches it opens, whose layout_mode
    is ``physics_3d`` or ``umap_3d``.
    """
    from utilities.Cache_Metadata import read_cache_metadata
    metadata = read_cache_metadata(path)
    # Checked first, and outside the try below: ViewerSettingsError is a
    # ValueError, which that block would re-wrap as a provenance failure.
    compatibility = (metadata.get("folder_manifest") or {}).get("compatibility") or {}
    layout_mode = compatibility.get("layout_mode")
    if not allow_3d and isinstance(layout_mode, str) and layout_mode.endswith("_3d"):
        raise ViewerSettingsError(
            f"{os.path.basename(str(path))} is a 3D layout cache (layout_mode "
            f"'{layout_mode}'). The desktop Viewer opens 2D caches only; 3D caches "
            "are for the optional VR viewer (opt_vr). Select this network's 2D "
            "cache, or generate one with LAYOUT_DIMENSIONS=2."
        )
    if metadata["status"] != "complete":
        raise ViewerSettingsError("Cache provenance: " + "; ".join(metadata["diagnostics"]))
    try:
        compatibility = metadata["folder_manifest"]["compatibility"]
        parameters = metadata["generation_parameters"]
        mode = compatibility["layout_mode"]
        if allow_3d and isinstance(mode, str) and mode.endswith("_3d"):
            mode = mode[: -len("_3d")]
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


def _path_resolver(directories, root):
    """The function that resolves one Config path: an alias is expanded from ``directories``, the rest taken from ``base`` or ``root``."""
    def resolve(value, base=None):
        value = os.path.expandvars(os.path.expanduser(value.strip()))
        for alias, key in ALIASES.items():
            if value == alias or value.startswith((alias + "/", alias + "\\")):
                value = os.path.join(directories[key], value[len(alias):].lstrip("/\\"))
                break
        if any(alias in value for alias in ALIASES):
            raise ViewerSettingsError(f"Invalid directory alias: {value}")
        return os.path.abspath(os.path.join(base or root, value))
    return resolve


def normalize_viewer_paths(values, project_root, *, require_cache=True):
    """Resolve Config paths without validating unrelated display/physics controls."""
    result = deepcopy(values)
    root = os.path.abspath(project_root)
    resolve = _path_resolver(result, root)

    for key in ALIASES.values():
        result[key] = resolve(result[key])
    for key in (*DIRECTORY_PROFILE_DEFAULTS, "SAVED_CONFIG_DIR"):
        if key not in ALIASES.values():
            result[key] = resolve(result[key])
    for key, directory in FILES.items():
        value = result[key].strip()
        if not value:
            if key not in OPTIONAL_FILES and (key != "TARGET_CACHE_PATH" or require_cache):
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


def _path_parts(path):
    """An absolute path as its drive and folder names, and each of them as the platform compares it (case, slashes)."""
    drive, tail = os.path.splitdrive(os.path.abspath(path))
    names = [drive, *(name for name in tail.split(os.sep) if name)]
    return names, [os.path.normcase(name) for name in names]


def alias_form(path, directories, project_root):
    """``path`` as the Config's Directories tab writes it: under the deepest folder it is in, as that folder's alias.

    The other way round from resolving: a file in the Networks_EValues folder
    of INPUT_FILE_DIR is "$input_file$/Networks_EValues/x.h5". ``directories``
    holds the four base folders by setting key (INPUT_FILE_DIR, CACHE_FILE_DIR,
    ANALYSIS_RESULT_DIR, EMBED_DIR), as the settings do: absolute, or an alias
    or a folder in ``project_root``, resolved as normalize_viewer_paths resolves
    them. A base that is missing or cannot be resolved is left out. The deepest
    base the path is under names it (the first in ALIASES order when two are
    one folder), the case and the slash ignored where the platform ignores
    them. Under no base, a path in ``project_root`` is written relative to it,
    and any other stays absolute. What follows the alias or the root is written
    with "/", as the tab writes it. No path is "".
    """
    if not path:
        return ""
    names, keys = _path_parts(path)
    best = None  # (names in the base, its alias)
    directories = directories or {}
    values = {key: directories.get(key) for key in ALIASES.values()}
    resolve = _path_resolver(values, os.path.abspath(project_root) if project_root else os.getcwd())
    for alias, key in ALIASES.items():
        try:
            values[key] = resolve(str(values[key])) if values[key] else None
        except (KeyError, TypeError, ViewerSettingsError):
            values[key] = None
        if values[key] is None:
            continue
        base = _path_parts(values[key])[1]
        if keys[:len(base)] == base and (best is None or len(base) > best[0]):
            best = (len(base), alias)
    if best is not None:
        return "/".join([best[1], *names[best[0]:]])
    if project_root:
        base = _path_parts(project_root)[1]
        if keys[:len(base)] == base and len(keys) > len(base):
            return "/".join(names[len(base):])
    return os.path.normpath(os.path.abspath(path))


def normalize_viewer_settings(document, project_root, *, require_cache=True):
    if not isinstance(document, dict):
        raise ViewerSettingsError("settings_document: expected a JSON object.")
    document = migrate_low_resource_mode(document)
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
        if result["EMBED_FILE"]:
            # The file's own interface only: its headers, sequences, model and
            # width. Embedding_Manager matches it to the nodes when it loads,
            # and partial coverage is allowed, as for the MSA.
            from Embedding_Manager import read_embedding_summary
            try:
                read_embedding_summary(result["EMBED_FILE"])
            except (OSError, ValueError, KeyError) as error:
                raise ViewerSettingsError(f"EMBED_FILE: {error}") from error
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

def resolve_selected_cache(settings, *, layout_dimensions=None):
    """Return the path of the layout cache that ``settings`` selects.

    An explicit TARGET_CACHE_PATH wins. Otherwise the folder is the canonical
    name of the inputs, and LAYOUT_DIMENSIONS - from ``settings`` unless
    ``layout_dimensions`` is given - picks the 2D folder or its ``_3D``
    sibling. Desktop settings carry no LAYOUT_DIMENSIONS, so the desktop
    viewer always resolves the 2D folder.

    The alignment reference plays no part in cache selection, so it is not
    resolved here. The alignment resolves it when it loads; read
    ``Alignment_Manager.resolved_ref_full`` for the header it anchored on.
    """
    if layout_dimensions is None:
        layout_dimensions = getattr(settings, "LAYOUT_DIMENSIONS", None) or 2
    saved_layout_dir = getattr(
        settings,
        "SAVED_LAYOUT_DIR",
        os.path.join("Cache_Files", "Saved_Layouts"),
    )

    explicit_path = getattr(settings, "TARGET_CACHE_PATH", None)
    if explicit_path:
        if os.path.isabs(explicit_path):
            return os.path.abspath(explicit_path)
        return cache_manifest.resolve_relative_cache_path(
            saved_layout_dir, explicit_path
        )

    fasta_file = getattr(settings, "NODE_FASTA_FILE", None) or getattr(
        settings, "SEQUENCES_FILE", ""
    )
    network_file = getattr(settings, "INPUT_HDF5", "")
    network_type = cache_manifest.validate_network_schema(network_file).network_type
    settings.INPUT_IS_EVALUE = network_type == "blast"
    name_settings = dict(
        alignment_score=getattr(settings, "ALIGNMENT_SCORE", None),
        normalization=getattr(settings, "NORM_MODE", None),
        umap_mode=getattr(settings, "UMAP_MODE", False),
        umap_neighbors=getattr(settings, "UMAP_NEIGHBORS", 15),
        top_edge_percent=getattr(settings, "TOP_EDGE_PERCENT", None),
        similarity_threshold=getattr(settings, "SIMILARITY_THRESHOLD", None),
        layout_dimensions=layout_dimensions,
    )
    canonical_name = cache_manifest.build_canonical_cache_name(
        fasta_file, network_file, network_type, **name_settings
    )
    target_folder = os.path.join(saved_layout_dir, canonical_name)
    if network_type == "blast" and not os.path.isdir(target_folder):
        # Caches of DIAMOND networks made before their folders were labelled
        # [DIAMOND] live under the model_name label, [BLAST].
        legacy_name = cache_manifest.build_canonical_cache_name(
            fasta_file, network_file, network_type, legacy_model_label=True, **name_settings
        )
        legacy_folder = os.path.join(saved_layout_dir, legacy_name)
        if os.path.isdir(legacy_folder):
            target_folder = legacy_folder

    selected_cache = getattr(settings, "TARGET_CACHE_FILE", None)
    if (
        isinstance(selected_cache, str)
        and selected_cache.strip()
        and selected_cache != "None"
    ):
        cache_manifest.validate_cache_filename(selected_cache)
        return os.path.join(target_folder, selected_cache)

    return os.path.join(target_folder, "version_00.h5")


# =====================================================================
# 5. Network Preparation (Filtering, Thresholding, UMAP)
# =====================================================================

_SEQUENCE_LENGTH_NORMALIZATIONS = frozenset(
    {"shorter_sequence", "longer_sequence", "average_sequence"}
)


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

    # Divide only where the denominator is positive and zero the rest, as
    # np.where(denominator > 0, raw_score / denominator, 0.0) did, without
    # holding its full quotient and its result at once (one score-sized
    # array per pair, ~320 MiB at 83 million pairs). The explicit out=None
    # tells NumPy the skipped pairs are filled below, which it otherwise
    # warns about.
    positive = denominator > 0
    scores = np.divide(raw_score, denominator, out=None, where=positive)
    skipped = np.logical_not(positive, out=positive)  # reuses the mask's memory
    scores[skipped] = 0.0
    return scores


def _read_pair_scores(data, settings, pair_mask, valid_sources, valid_targets):
    """Return the normalized score of each pair that ``pair_mask`` keeps.

    Reads only the per-pair columns the settings use: the alignment-length
    column only when the normalization divides by it, and never the other
    score type's pair. Its own frame releases the full-length columns before
    the caller ranks and thresholds the scores.
    """
    if settings.INPUT_IS_EVALUE:
        return data["score"][:][pair_mask]

    if settings.ALIGNMENT_SCORE == "global":
        score_name, length_name = "g_score", "g_len"
    else:
        score_name, length_name = "l_score", "l_len"
    raw_scores = data[score_name][:][pair_mask]
    alignment_lengths = source_lengths = target_lengths = None
    if settings.NORM_MODE in _SEQUENCE_LENGTH_NORMALIZATIONS:
        sequence_lengths = data["seq_lens"][:]
        source_lengths = sequence_lengths[valid_sources]
        target_lengths = sequence_lengths[valid_targets]
    else:
        alignment_lengths = data[length_name][:][pair_mask]
    return _normalize_score(
        raw_scores,
        alignment_lengths,
        source_lengths,
        target_lengths,
        settings.NORM_MODE,
    )


def _strongest_pairs(data, settings, *, kept_mask, keep_limit):
    """The pairs that are among the best ``keep_limit`` of their source or of their target.

    Returns the stored node indices of each pair and its score, in the order of
    the stored pairs, streaming the network instead of loading its columns. A
    node's best pairs are the highest scores; equal scores go to the pair stored
    first. A subset of nodes (``kept_mask``) ranks only the pairs between its
    nodes, as the network's columns would be filtered before ranking.
    """
    from utilities import Network_Score_Statistics as scan
    from utilities import Network_Top_Edges as top

    total_nodes = len(kept_mask)
    all_kept = bool(kept_mask.all())
    score_mode, norm_mode = "global", "alignment_length"
    if not settings.INPUT_IS_EVALUE:  # E-value scores are stored as they are used
        score_mode = "global" if settings.ALIGNMENT_SCORE == "global" else "local"
        norm_mode = settings.NORM_MODE
    source = scan.ScoreSource(
        data,
        is_blast=settings.INPUT_IS_EVALUE,
        score_mode=score_mode,
        norm_mode=norm_mode,
        kept_mask=None if all_kept else kept_mask,
        # This module's own normalization, on the columns as stored.
        normalize=_normalize_score,
        cast_float32=False,
        # With every node kept, a pair index past the header count is an error
        # that no mask would catch; with a subset, the mask's lookup raises it.
        node_count=total_nodes if all_kept else None,
    )
    _, sources, targets, scores = top.strongest_edges(source, total_nodes, keep_limit)
    return sources, targets, scores


def prepare_network(data, *, settings, selected_fasta_headers):
    """Return filtered ``(headers, edges, scores)`` for an open network file.

    Only nodes named in ``selected_fasta_headers`` (the selected FASTA's
    sanitized headers) are kept. In UMAP mode only each node's best
    ``UMAP_NEIGHBORS`` edges, as a source and as a target, are returned.
    """
    metadata = cache_manifest.validate_network_schema(data)
    settings.INPUT_IS_EVALUE = metadata.network_type == "blast"

    raw_headers = data["headers"][:]
    headers = [
        header.decode("utf-8") if isinstance(header, bytes) else header
        for header in raw_headers
    ]
    total_nodes = len(headers)

    print(f"Raw Data: {total_nodes} sequences.")
    if not settings.INPUT_IS_EVALUE:
        print(
            f"Metric: {settings.ALIGNMENT_SCORE.upper()} Alignment with "
            f"{settings.NORM_MODE} Normalization"
        )

    fasta_path = getattr(settings, "NODE_FASTA_FILE", "")
    kept_indices = []
    clean_fasta_path = os.path.normpath(fasta_path)
    print(f"Scanning FASTA file for node filter: {clean_fasta_path}")
    fasta_ids = set()
    fasta_headers = set()
    try:
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

    kept_mask = np.zeros(total_nodes, dtype=bool)
    kept_mask[kept_indices] = True
    filtered_headers = [headers[index] for index in kept_indices]

    index_map = np.zeros(total_nodes, dtype=np.int32)
    index_map[kept_indices] = np.arange(len(kept_indices))

    if getattr(settings, "UMAP_MODE", False):
        # The pairs are streamed: a network can hold close to a billion of them, and
        # their columns, a sorted copy and the group codes need far more memory.
        print(
            "UMAP Mode enabled: Bypassing global threshold. Filtering top k "
            "edges per node..."
        )
        keep_limit = int(getattr(settings, "UMAP_NEIGHBORS", 15))
        edge_sources, edge_targets, edge_scores = _strongest_pairs(
            data, settings, kept_mask=kept_mask, keep_limit=keep_limit
        )
        print(
            f"Kept {len(edge_scores)} edges for UMAP topology "
            f"(max {keep_limit} per node direction)."
        )
        edges = np.column_stack(
            (index_map[edge_sources], index_map[edge_targets])
        ).astype(np.int32)
        return filtered_headers, edges, edge_scores

    sources = data["i"][:]
    targets = data["j"][:]
    if kept_mask.all():
        # Every pair is valid, so skip the pair mask and its full-length
        # copies (seconds on networks with tens of millions of pairs). The
        # mask's indexing was also the bounds check, so keep that explicitly.
        if len(sources) and max(int(sources.max()), int(targets.max())) >= total_nodes:
            raise IndexError("Network pair indices exceed the header count.")
        valid_edges_mask = slice(None)
    else:
        valid_edges_mask = kept_mask[sources] & kept_mask[targets]
    valid_sources = sources[valid_edges_mask]
    valid_targets = targets[valid_edges_mask]
    valid_scores = _read_pair_scores(
        data, settings, valid_edges_mask, valid_sources, valid_targets
    )

    top_percent = getattr(settings, "TOP_EDGE_PERCENT", None)
    if top_percent is not None:
        active_nodes = len(kept_indices)
        theoretical_max_edges = (active_nodes * (active_nodes - 1)) / 2.0
        edge_count = int(theoretical_max_edges * (top_percent / 100.0))
        if len(valid_scores) == 0:
            calculated_cutoff = 0.0
        else:
            edge_count = max(1, min(edge_count, len(valid_scores)))
            # The edge_count-th largest score in linear time. np.partition
            # shares np.sort's ordering (NaN last), so this is exactly
            # np.sort(valid_scores)[::-1][edge_count - 1], without the sort.
            kth = len(valid_scores) - edge_count
            calculated_cutoff = float(np.partition(valid_scores, kth)[kth])

        mode_label = "E-Value" if settings.INPUT_IS_EVALUE else "Similarity"
        print(
            f"Top {top_percent}% Edges Requested (based on max possible "
            f"{int(theoretical_max_edges)} edges)."
        )
        print(f"Calculated {mode_label} Cutoff: {calculated_cutoff:.5f}")
        settings.SIMILARITY_THRESHOLD = calculated_cutoff

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
    "EMBEDDING_ALIAS",
    "INPUT_PROFILE_DEFAULTS",
    "VISUAL_PROFILE_DEFAULTS",
    "LOW_RESOURCE_DEFAULTS",
    "LEGACY_LOW_RESOURCE_KEY",
    "migrate_low_resource_mode",
    "fewer_edges_while_moving",
    "low_resource_animations",
    "low_resource_drawing",
    "low_resource_webserver",
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
    "OPTIONAL_FILES",
    "get_viewer_settings_schema",
    "resolve_cache_settings",
    "resolve_viewer_document",
    "validate_viewer_document",
    "read_viewer_settings",
    "normalize_viewer_settings",
    "normalize_viewer_paths",
    "alias_form",
    "resolve_selected_cache",
    "prepare_network",
]
