"""Versioned execution documents; personal GUI profiles remain flat."""
from copy import deepcopy
from utilities.Viewer_Defaults import VISUAL_PROFILE_DEFAULTS, DIRECTORY_PROFILE_DEFAULTS

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
    return {"schema_version": 2, "kind": kind,
            **{section: {key: deepcopy(values[key]) for key in keys}
               for section, keys in sections(kind).items()}}


def decode_document(document, kind, *, partial=False):
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document.get("schema_version") != 2 or document.get("kind") != kind:
        raise ValueError(f"Expected schema_version 2, kind '{kind}'. Re-export settings through GUI or export_config_settings; legacy execution JSON is unsupported.")
    mapping = sections(kind)
    unknown = set(document) - {"schema_version", "kind", *mapping}
    if unknown:
        raise ValueError("Unknown document sections: " + ", ".join(sorted(unknown)))
    result = {}
    for section, keys in mapping.items():
        values = document.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section}: expected an object.")
        unknown = set(values) - set(keys)
        missing = set(keys) - set(values)
        if unknown:
            raise ValueError(f"{section}: unknown or misplaced fields: " + ", ".join(sorted(unknown)))
        if missing and not partial:
            raise ValueError(f"{section}: missing fields: " + ", ".join(sorted(missing)))
        result.update(deepcopy(values))
    return result
