# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Read cache provenance without loading coordinate or network datasets."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os


def _generation_parameters(attrs):
    parameters = json.loads(attrs["layout_compatibility_json"])
    if not isinstance(parameters, dict):
        raise ValueError("layout_compatibility_json must contain an object")
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if "layout_compatibility_id" in attrs and hashlib.sha256(canonical.encode("utf-8")).hexdigest() != attrs["layout_compatibility_id"]:
        raise ValueError("Layout parameter hash does not match layout_compatibility_id")
    return parameters


def validate_cache_provenance(attributes, manifest_id):
    """Return a detached, validated copy of the original HDF5 provenance."""
    if attributes is None:
        raise ValueError("The active viewer has no cache provenance binding.")
    result = {}
    for name in ("cache_manifest_id", "layout_compatibility_json", "layout_compatibility_id"):
        if name not in attributes:
            raise ValueError(f"Missing HDF5 attribute: {name}")
        value = attributes[name]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str):
            raise ValueError(f"{name} must be text")
        result[name] = value
    _generation_parameters(result)
    if result["cache_manifest_id"] != manifest_id:
        raise ValueError("Cache provenance manifest ID differs from active manifest")
    return result


def _signature(path):
    try:
        stat = os.stat(path)
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except OSError:
        return None


def read_cache_metadata(path):
    if not isinstance(path, str) or not path.strip():
        return {"cache_path": None, "cache_filename": None, "status": "unavailable",
                "diagnostics": ["Viewer did not report a cache path."], "attributes": {},
                "generation_parameters": None, "folder_manifest": None}
    path = os.path.abspath(path)
    manifest = os.path.join(os.path.dirname(path), "cache_manifest.json")
    return deepcopy(_read(path, _signature(path), _signature(manifest)))


@lru_cache(maxsize=128)
def _read(path, cache_signature, manifest_signature):
    import h5py
    import Cache_Manifest
    result = {"cache_path": path, "cache_filename": os.path.basename(path),
              "status": "complete", "diagnostics": [], "attributes": {},
              "generation_parameters": None, "folder_manifest": None}
    errors = result["diagnostics"]
    invalid = False
    try:
        with h5py.File(path, "r") as cache:
            for name in ("cache_manifest_id", "layout_compatibility_json", "layout_compatibility_id"):
                if name not in cache.attrs:
                    errors.append(f"Missing HDF5 attribute: {name}")
                    continue
                value = cache.attrs[name]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be text")
                result["attributes"][name] = value
        attrs = result["attributes"]
        if "layout_compatibility_json" in attrs:
            result["generation_parameters"] = _generation_parameters(attrs)
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"Cache attributes: {error}")
        invalid = cache_signature is not None
    try:
        manifest = Cache_Manifest.read_manifest(os.path.dirname(path))
        result["folder_manifest"] = manifest
        recorded = result["attributes"].get("cache_manifest_id")
        if recorded and recorded != manifest["manifest_id"]:
            raise ValueError("Cache manifest ID differs from parent folder manifest")
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(f"Folder manifest: {error}")
        invalid = invalid or manifest_signature is not None
    result["status"] = "invalid" if invalid else "partial" if errors else "complete"
    return result


__all__ = ["read_cache_metadata", "validate_cache_provenance", "_signature", "_read"]
