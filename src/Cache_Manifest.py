"""Cache-folder manifest helpers for SSN layout discovery and validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone


MANIFEST_FILENAME = "cache_manifest.json"
HASH_CHUNK_SIZE = 8 * 1024 * 1024


class CacheManifestError(ValueError):
    """Raised when cache identity or path data is unsafe or malformed."""


class NetworkMetadataError(CacheManifestError):
    """Raised when required HDF5 network metadata or schema is invalid."""


class CacheHashCancelled(RuntimeError):
    """Raised when a background input hash is no longer needed."""


@dataclass(frozen=True)
class NetworkMetadata:
    """Normalized metadata used to select the network loading behavior."""

    model_name: str
    network_type: str


@dataclass(frozen=True)
class NetworkCompletenessInfo:
    """Constant-time HDF5 network completeness metadata."""

    status: str
    sequence_count: int = 0
    edge_count: int = 0
    expected_edge_count: int = 0
    reason: str = ""


def inspect_network_completeness(network_path):
    """Classify a network using HDF5 dataset shapes without reading edge data."""
    import h5py

    resolved = os.path.abspath(os.path.normpath(network_path))
    try:
        with h5py.File(resolved, "r") as network:
            missing = [name for name in ("headers", "i", "j") if name not in network]
            if missing:
                return NetworkCompletenessInfo(
                    status="unknown",
                    reason=f"Missing required dataset(s): {', '.join(missing)}.",
                )

            headers = network["headers"]
            edge_i = network["i"]
            edge_j = network["j"]
            if headers.ndim != 1 or edge_i.ndim != 1 or edge_j.ndim != 1:
                return NetworkCompletenessInfo(
                    status="unknown",
                    reason="The headers, i, and j datasets must all be one-dimensional.",
                )

            sequence_count = int(headers.shape[0])
            edge_count_i = int(edge_i.shape[0])
            edge_count_j = int(edge_j.shape[0])
            expected_edge_count = sequence_count * (sequence_count - 1) // 2

            if edge_count_i != edge_count_j:
                return NetworkCompletenessInfo(
                    status="unknown",
                    sequence_count=sequence_count,
                    edge_count=min(edge_count_i, edge_count_j),
                    expected_edge_count=expected_edge_count,
                    reason=(
                        "The i and j edge datasets have different lengths "
                        f"({edge_count_i:,} and {edge_count_j:,})."
                    ),
                )

            if edge_count_i > expected_edge_count:
                return NetworkCompletenessInfo(
                    status="unknown",
                    sequence_count=sequence_count,
                    edge_count=edge_count_i,
                    expected_edge_count=expected_edge_count,
                    reason=(
                        f"The network contains {edge_count_i:,} edges, exceeding the "
                        f"{expected_edge_count:,} unique undirected pairs expected."
                    ),
                )

            status = "complete" if edge_count_i == expected_edge_count else "incomplete"
            return NetworkCompletenessInfo(
                status=status,
                sequence_count=sequence_count,
                edge_count=edge_count_i,
                expected_edge_count=expected_edge_count,
            )
    except (OSError, ValueError, KeyError) as error:
        return NetworkCompletenessInfo(
            status="unknown",
            reason=f"Unable to inspect network metadata: {error}",
        )


def calculate_file_sha256(
    file_path, chunk_size=HASH_CHUNK_SIZE, cancellation_requested=None
):
    """Return a SHA-256 digest without loading the complete file into memory."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            if cancellation_requested and cancellation_requested():
                raise CacheHashCancelled("Input hashing was cancelled.")
            hasher.update(chunk)
    return hasher.hexdigest()


def fingerprint_file(file_path, cancellation_requested=None):
    """Return the informational and compatibility identity of one input file."""
    resolved = os.path.abspath(os.path.normpath(file_path))
    stat = os.stat(resolved)
    return {
        "basename": os.path.basename(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": calculate_file_sha256(
            resolved, cancellation_requested=cancellation_requested
        ),
    }


def file_cache_key(file_path):
    """Return the in-process key used to reuse a completed GUI hash."""
    resolved = os.path.abspath(os.path.normpath(file_path))
    stat = os.stat(resolved)
    return resolved, int(stat.st_size), int(stat.st_mtime_ns)


def _network_source_label(network_source):
    filename = getattr(network_source, "filename", None)
    if filename:
        return os.path.abspath(os.path.normpath(os.fspath(filename)))
    try:
        return os.path.abspath(os.path.normpath(os.fspath(network_source)))
    except TypeError:
        return "<open HDF5 network>"


def _read_open_network_metadata(network, source_label):
    if "model_name" not in network.attrs:
        raise NetworkMetadataError(
            f"Network file '{source_label}' is missing the required root attribute "
            "'model_name'. Regenerate the network with a supported SSN tool or add "
            "valid model metadata before loading it."
        )

    raw_model_name = network.attrs["model_name"]
    if isinstance(raw_model_name, bytes):
        try:
            model_name = raw_model_name.decode("utf-8")
        except UnicodeDecodeError as error:
            raise NetworkMetadataError(
                f"Network file '{source_label}' has a 'model_name' attribute that "
                "is not valid UTF-8."
            ) from error
    elif isinstance(raw_model_name, str):
        model_name = raw_model_name
    else:
        raise NetworkMetadataError(
            f"Network file '{source_label}' has a non-text 'model_name' attribute "
            f"of type {type(raw_model_name).__name__}."
        )

    model_name = model_name.strip()
    if not model_name:
        raise NetworkMetadataError(
            f"Network file '{source_label}' has a blank required root attribute "
            "'model_name'."
        )

    network_type = "blast" if model_name.casefold() == "blast" else "alignment"
    return NetworkMetadata(model_name=model_name, network_type=network_type)


def read_network_metadata(network_source):
    """Read normalized network metadata from a path or open HDF5 file."""
    import h5py

    source_label = _network_source_label(network_source)
    if hasattr(network_source, "attrs"):
        return _read_open_network_metadata(network_source, source_label)

    try:
        network_path = os.fspath(network_source)
    except TypeError as error:
        raise TypeError(
            "network_source must be a filesystem path or an open HDF5 file"
        ) from error

    with h5py.File(network_path, "r") as network:
        return _read_open_network_metadata(network, source_label)


def detect_network_type(network_source):
    """Return ``blast`` or ``alignment`` using only ``model_name`` metadata."""
    return read_network_metadata(network_source).network_type


def validate_network_schema(network_source, expected_network_type=None):
    """Validate datasets required by the metadata-selected network type."""
    import h5py

    source_label = _network_source_label(network_source)
    if not hasattr(network_source, "attrs"):
        try:
            network_path = os.fspath(network_source)
        except TypeError as error:
            raise TypeError(
                "network_source must be a filesystem path or an open HDF5 file"
            ) from error
        with h5py.File(network_path, "r") as network:
            return validate_network_schema(network, expected_network_type)

    metadata = read_network_metadata(network_source)
    if (
        expected_network_type is not None
        and expected_network_type != metadata.network_type
    ):
        raise NetworkMetadataError(
            f"Network file '{source_label}' declares network type "
            f"'{metadata.network_type}' through model_name='{metadata.model_name}', "
            f"not the expected type '{expected_network_type}'."
        )

    required_datasets = {
        "blast": ("headers", "i", "j", "score"),
        "alignment": (
            "headers",
            "seq_lens",
            "i",
            "j",
            "l_score",
            "l_len",
            "g_score",
            "g_len",
        ),
    }[metadata.network_type]
    missing = [name for name in required_datasets if name not in network_source]
    if missing:
        raise NetworkMetadataError(
            f"Network file '{source_label}' declares model_name="
            f"'{metadata.model_name}' ({metadata.network_type}) but is missing "
            f"required dataset(s): {', '.join(missing)}."
        )
    return metadata


def _optional_float(value):
    if value is None or str(value).strip() in {"", "None"}:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CacheManifestError("Cache compatibility values must be finite.")
    return parsed


def build_compatibility(
    sequence_sha256,
    network_sha256,
    network_type,
    alignment_score=None,
    normalization=None,
    umap_mode=False,
    umap_neighbors=15,
    top_edge_percent=None,
    similarity_threshold=None,
):
    """Build the canonical fields that define one compatible cache folder."""
    network_type = str(network_type).lower()
    if network_type not in {"blast", "alignment"}:
        raise CacheManifestError(f"Unsupported network type: {network_type}")

    is_umap = bool(umap_mode)
    if is_umap:
        edge_filter = {"mode": "umap_neighbors", "value": int(umap_neighbors)}
    else:
        top_value = _optional_float(top_edge_percent)
        if top_value is not None:
            edge_filter = {"mode": "top_edge_percent", "value": top_value}
        else:
            edge_filter = {
                "mode": "similarity_threshold",
                "value": _optional_float(similarity_threshold),
            }

    return {
        "sequence_sha256": str(sequence_sha256).lower(),
        "network_sha256": str(network_sha256).lower(),
        "network_type": network_type,
        "alignment_score": None if network_type == "blast" else str(alignment_score),
        "normalization": None if network_type == "blast" else str(normalization),
        "layout_mode": "umap" if is_umap else "physics",
        "edge_filter": edge_filter,
    }


def calculate_manifest_id(compatibility):
    """Hash canonical compatibility JSON to bind cache files to a folder manifest."""
    canonical = json.dumps(
        compatibility,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(sequence_fingerprint, network_fingerprint, compatibility):
    """Create a complete, human-inspectable folder manifest."""
    return {
        "manifest_id": calculate_manifest_id(compatibility),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "sequence": dict(sequence_fingerprint),
            "network": dict(network_fingerprint),
        },
        "compatibility": compatibility,
    }


def build_manifest_for_files(sequence_path, network_path, **settings):
    """Hash both inputs and construct their current manifest."""
    sequence = fingerprint_file(sequence_path)
    network = fingerprint_file(network_path)
    network_type = validate_network_schema(network_path).network_type
    compatibility = build_compatibility(
        sequence["sha256"],
        network["sha256"],
        network_type,
        **settings,
    )
    return build_manifest(sequence, network, compatibility)


def validate_manifest(manifest, expected_compatibility=None):
    """Validate required fields and optionally compare current compatibility."""
    if not isinstance(manifest, dict):
        raise CacheManifestError("Manifest root must be a JSON object.")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, dict):
        raise CacheManifestError("Manifest compatibility must be a JSON object.")
    manifest_id = manifest.get("manifest_id")
    calculated_id = calculate_manifest_id(compatibility)
    if manifest_id != calculated_id:
        raise CacheManifestError("Manifest ID does not match its compatibility fields.")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise CacheManifestError("Manifest inputs must be a JSON object.")
    for key, digest_key in (("sequence", "sequence_sha256"), ("network", "network_sha256")):
        record = inputs.get(key)
        if not isinstance(record, dict):
            raise CacheManifestError(f"Manifest input '{key}' is missing.")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise CacheManifestError(f"Manifest input '{key}' has an invalid SHA-256.")
        if digest.lower() != compatibility.get(digest_key):
            raise CacheManifestError(f"Manifest input '{key}' conflicts with compatibility.")

    if expected_compatibility is not None and compatibility != expected_compatibility:
        raise CacheManifestError("Manifest does not match the current inputs and settings.")
    return manifest


def read_manifest(folder_path, expected_compatibility=None):
    manifest_path = os.path.join(folder_path, MANIFEST_FILENAME)
    with open(manifest_path, "r", encoding="utf-8") as source:
        manifest = json.load(source)
    return validate_manifest(manifest, expected_compatibility)


def write_manifest_atomic(folder_path, manifest):
    """Validate and atomically publish a folder manifest."""
    validate_manifest(manifest)
    os.makedirs(folder_path, exist_ok=True)
    final_path = os.path.join(folder_path, MANIFEST_FILENAME)
    partial_path = final_path + ".partial"
    try:
        with open(partial_path, "w", encoding="utf-8", newline="\n") as output:
            json.dump(manifest, output, indent=2, sort_keys=True, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial_path, final_path)
    finally:
        if os.path.exists(partial_path):
            os.remove(partial_path)
    return final_path


def copy_file_atomic(source_path, destination_path):
    """Copy one file without ever exposing a partially written destination."""
    source = os.path.abspath(os.path.normpath(source_path))
    destination = os.path.abspath(os.path.normpath(destination_path))
    if not os.path.isfile(source):
        raise CacheManifestError(f"Backup source file does not exist: {source}")
    if os.path.normcase(source) == os.path.normcase(destination):
        raise CacheManifestError("Backup source and destination must be different files.")
    if os.path.exists(destination):
        raise CacheManifestError(f"Backup destination already exists: {destination}")

    partial_path = destination + ".partial"
    try:
        shutil.copy2(source, partial_path)
        os.replace(partial_path, destination)
    finally:
        if os.path.exists(partial_path):
            os.remove(partial_path)
    return destination


def _is_within(root_path, candidate_path):
    root = os.path.abspath(os.path.realpath(root_path))
    candidate = os.path.abspath(os.path.realpath(candidate_path))
    try:
        return os.path.commonpath([root, candidate]) == root
    except ValueError:
        return False


def find_matching_manifest_folders(saved_layout_dir, expected_compatibility):
    """Scan immediate safe child folders and return matching manifest records."""
    root = os.path.abspath(os.path.normpath(saved_layout_dir))
    if not os.path.isdir(root):
        return []

    matches = []
    for entry in os.scandir(root):
        if not entry.is_dir(follow_symlinks=False):
            continue
        folder = os.path.abspath(entry.path)
        if not _is_within(root, folder):
            continue
        manifest_path = os.path.join(folder, MANIFEST_FILENAME)
        if not os.path.isfile(manifest_path):
            continue
        try:
            manifest = read_manifest(folder, expected_compatibility)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        matches.append({"folder": folder, "manifest": manifest})
    return sorted(matches, key=lambda item: os.path.normcase(item["folder"]))


def build_canonical_cache_name(
    sequence_path,
    network_path,
    network_type,
    alignment_score=None,
    normalization=None,
    umap_mode=False,
    umap_neighbors=15,
    top_edge_percent=None,
    similarity_threshold=None,
):
    """Build a human-readable cache name from authoritative network metadata."""
    fasta_base = os.path.splitext(os.path.basename(sequence_path))[0] or "Network"
    metadata = validate_network_schema(
        network_path,
        expected_network_type=str(network_type).lower(),
    )
    model_label = re.sub(r'[<>:"/\\|?*]', "_", metadata.model_name)
    model_string = f"_[{model_label}]"

    suffix = ""
    if str(network_type).lower() != "blast":
        if normalization:
            suffix += f"_{normalization}"
        if alignment_score:
            suffix += f"_{alignment_score}"
    if umap_mode:
        suffix += f"_UMAP_k{int(umap_neighbors)}"
    else:
        top_value = _optional_float(top_edge_percent)
        if top_value is not None:
            suffix += f"_Top{top_value}Pct"
        else:
            threshold = _optional_float(similarity_threshold)
            if threshold is not None:
                suffix += f"_Score{threshold}"
    return f"{fasta_base}{model_string}{suffix}"


def next_cache_version_filename(folder_path):
    """Return the next simple two-digit default cache filename."""
    max_version = -1
    if os.path.isdir(folder_path):
        pattern = re.compile(r"^version_(\d+)\.h5$", re.IGNORECASE)
        for entry in os.scandir(folder_path):
            if not entry.is_file():
                continue
            match = pattern.fullmatch(entry.name)
            if match:
                max_version = max(max_version, int(match.group(1)))
    return f"version_{max_version + 1:02d}.h5"


def validate_cache_filename(filename):
    """Accept only one plain HDF5 basename."""
    if not isinstance(filename, str) or not filename:
        raise CacheManifestError("Cache filename is empty.")
    if os.path.isabs(filename) or filename != os.path.basename(filename):
        raise CacheManifestError("Cache filename must be a plain basename.")
    if filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise CacheManifestError("Cache filename cannot contain path components.")
    if not filename.lower().endswith(".h5"):
        raise CacheManifestError("Cache filename must end with .h5.")
    return filename


def relative_cache_path(saved_layout_dir, folder_path, filename):
    """Return a validated relative cache path for subprocess routing."""
    validate_cache_filename(filename)
    root = os.path.abspath(os.path.normpath(saved_layout_dir))
    folder = os.path.abspath(os.path.normpath(folder_path))
    if not _is_within(root, folder) or os.path.dirname(folder) != root:
        raise CacheManifestError("Cache folder must be an immediate child of SAVED_LAYOUT_DIR.")
    candidate = os.path.join(folder, filename)
    if not _is_within(root, candidate):
        raise CacheManifestError("Cache path escapes SAVED_LAYOUT_DIR.")
    return os.path.relpath(candidate, root)


def resolve_relative_cache_path(saved_layout_dir, relative_path):
    """Resolve a GUI-provided relative cache path without allowing traversal."""
    if not isinstance(relative_path, str) or not relative_path or os.path.isabs(relative_path):
        raise CacheManifestError("Cache path must be relative to SAVED_LAYOUT_DIR.")
    normalized = os.path.normpath(relative_path)
    parts = normalized.split(os.sep)
    if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
        raise CacheManifestError("Cache path must contain one folder and one filename.")
    validate_cache_filename(parts[1])
    root = os.path.abspath(os.path.normpath(saved_layout_dir))
    candidate = os.path.abspath(os.path.join(root, normalized))
    if not _is_within(root, candidate):
        raise CacheManifestError("Cache path escapes SAVED_LAYOUT_DIR.")
    return candidate


def validate_cache_hdf5(hf, expected_headers, manifest_id):
    """Validate one selected cache before any cached state is applied."""
    import numpy as np

    stored_id = hf.attrs.get("cache_manifest_id")
    if isinstance(stored_id, bytes):
        stored_id = stored_id.decode("utf-8", errors="replace")
    if stored_id != manifest_id:
        raise CacheManifestError("Cache file was not created for this folder manifest.")
    for required in ("headers", "positions"):
        if required not in hf:
            raise CacheManifestError(f"Cache file is missing required dataset '{required}'.")

    raw_headers = hf["headers"][:]
    cached_headers = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_headers
    ]
    if cached_headers != list(expected_headers):
        raise CacheManifestError("Cache headers do not match the current node order.")

    node_count = len(cached_headers)
    positions = hf["positions"][:]
    if positions.shape != (node_count, 2):
        raise CacheManifestError("Cache positions must have shape (node_count, 2).")
    if not np.issubdtype(positions.dtype, np.number) or not np.all(np.isfinite(positions)):
        raise CacheManifestError("Cache positions contain invalid coordinates.")

    for dataset_name in ("colors", "sizes", "shapes", "visible_mask", "cluster_labels"):
        if dataset_name in hf and len(hf[dataset_name]) != node_count:
            raise CacheManifestError(
                f"Cache dataset '{dataset_name}' does not match the node count."
            )
    if "metadata" in hf:
        for property_name, dataset in hf["metadata"].items():
            if len(dataset) != node_count:
                raise CacheManifestError(
                    f"Cache metadata '{property_name}' does not match the node count."
                )
    if "group_labels" in hf:
        raw_groups = hf["group_labels"][()]
        if isinstance(raw_groups, bytes):
            raw_groups = raw_groups.decode("utf-8")
        groups = json.loads(raw_groups)
        if not isinstance(groups, list) or len(groups) != node_count:
            raise CacheManifestError("Cache group labels do not match the node count.")
    return cached_headers, positions
