# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

"""Generate one SSN layout cache without starting the interactive viewer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping

import h5py
import numpy as np

import Cache_Manifest as cache_manifest
from utilities.FASTA_Sanitization import load_sanitized_fasta
from utilities.Network_Preparation import prepare_network


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = Path(__file__).name


class LayoutGenerationError(ValueError):
    """Raised when settings or cache publication are unsafe or invalid."""


_REQUIRED_JSON_KEYS = {
    "NODE_FASTA_FILE",
    "INPUT_HDF5",
    "CACHE_FILENAME",
    "ALIGNMENT_SCORE",
    "NORM_MODE",
    "SIMILARITY_THRESHOLD",
    "TOP_EDGE_PERCENT",
    "UMAP_MODE",
    "UMAP_NEIGHBORS",
    "UMAP_MIN_DIST",
    "PHYSICS_ENGINE",
    "LAYOUT_DEVICE_SELECTION",
    "SPRING_K",
    "COULOMB_K",
    "COULOMB_CUTOFF",
    "DAMPING",
    "DT",
    "MAX_STEPS",
    "RMSD_THRESHOLD",
    "PERCENTAGE_DROP_THRESHOLD",
    "RMSD_WINDOW",
    "ENABLE_PROGRESSIVE_SIMULATION",
    "PACKING_GEOMETRY",
    "PACKING_GRID_SIZE",
    "SGLD_MIN_K",
    "SGLD_K_PERCENT",
    "SGLD_START_TEMP",
    "SGLD_NOISE_SCALE",
}


def _resolve_project_path(value: Any, project_root: Path) -> str:
    raw_path = os.fspath(value).strip()
    if not raw_path:
        raise LayoutGenerationError("Layout generation paths cannot be empty.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return os.path.abspath(os.path.normpath(path))


def _portable_path(path: str, project_root: Path) -> str:
    absolute = Path(path).resolve()
    try:
        return absolute.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return os.fspath(absolute)


def _finite_number(name: str, value: Any, *, minimum=None, maximum=None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LayoutGenerationError(f"{name} must be a JSON number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise LayoutGenerationError(f"{name} must be finite.")
    if minimum is not None and numeric < minimum:
        raise LayoutGenerationError(f"{name} must be at least {minimum}.")
    if maximum is not None and numeric > maximum:
        raise LayoutGenerationError(f"{name} must be at most {maximum}.")


@dataclass
class LayoutGenerationSettings:
    """Validated, viewer-independent inputs for one cache generation run."""

    SAVED_LAYOUT_DIR: str
    NODE_FASTA_FILE: str
    INPUT_HDF5: str
    CACHE_FILENAME: str

    ALIGNMENT_SCORE: str | None
    NORM_MODE: str | None
    SIMILARITY_THRESHOLD: float | None
    TOP_EDGE_PERCENT: float | None

    UMAP_MODE: bool
    UMAP_NEIGHBORS: int
    UMAP_MIN_DIST: float

    PHYSICS_ENGINE: str
    LAYOUT_DEVICE_SELECTION: str
    SPRING_K: float
    COULOMB_K: float
    COULOMB_CUTOFF: float
    DAMPING: float
    DT: float
    MAX_STEPS: int
    RMSD_THRESHOLD: float
    PERCENTAGE_DROP_THRESHOLD: float
    RMSD_WINDOW: int
    ENABLE_PROGRESSIVE_SIMULATION: bool
    PACKING_GEOMETRY: str
    PACKING_GRID_SIZE: float
    SGLD_MIN_K: int
    SGLD_K_PERCENT: float
    SGLD_START_TEMP: float
    SGLD_NOISE_SCALE: float

    # Coordinate-affecting values which are currently hidden in EMAP-SSN Configuration.
    BOX_SCALE: float = 2.0
    PACKING_PADDING: float = 10.0
    MAX_FORCE_LIMIT: float = 20.0
    MAX_TOTAL_REPULSION_FORCE: float = 0.0

    # Viewer launches may bind a GUI-resolved folder. This field is never JSON.
    _target_cache_path: str | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
        *,
        project_root: str | os.PathLike[str] = PROJECT_ROOT,
    ) -> "LayoutGenerationSettings":
        if not isinstance(document, Mapping):
            raise LayoutGenerationError("The settings document must be a JSON object.")
        if set(document) != {"DIRECTORIES", SCRIPT_NAME}:
            raise LayoutGenerationError(
                f"The settings document must contain only DIRECTORIES and {SCRIPT_NAME}."
            )
        directories = document["DIRECTORIES"]
        values = document[SCRIPT_NAME]
        if not isinstance(directories, Mapping) or set(directories) != {"SAVED_LAYOUT_DIR"}:
            raise LayoutGenerationError(
                "DIRECTORIES must contain only SAVED_LAYOUT_DIR."
            )
        if not isinstance(values, Mapping):
            raise LayoutGenerationError(f"{SCRIPT_NAME} must be a JSON object.")

        allowed = {item.name for item in fields(cls) if not item.name.startswith("_")}
        unknown = sorted(set(values) - allowed)
        missing = sorted(_REQUIRED_JSON_KEYS - set(values))
        if unknown:
            raise LayoutGenerationError(
                "Unknown layout-generation setting(s): " + ", ".join(unknown)
            )
        if missing:
            raise LayoutGenerationError(
                "Missing layout-generation setting(s): " + ", ".join(missing)
            )

        root = Path(project_root).resolve()
        payload = dict(values)
        payload["SAVED_LAYOUT_DIR"] = _resolve_project_path(
            directories["SAVED_LAYOUT_DIR"], root
        )
        payload["NODE_FASTA_FILE"] = _resolve_project_path(
            payload["NODE_FASTA_FILE"], root
        )
        payload["INPUT_HDF5"] = _resolve_project_path(payload["INPUT_HDF5"], root)
        settings = cls(**payload)
        settings.validate()
        return settings

    @classmethod
    def from_json_file(
        cls,
        settings_path: str | os.PathLike[str],
        *,
        project_root: str | os.PathLike[str] = PROJECT_ROOT,
    ) -> "LayoutGenerationSettings":
        path = Path(settings_path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise LayoutGenerationError(
                f"Could not read layout settings '{path}': {error}"
            ) from error
        return cls.from_document(document, project_root=project_root)

    @classmethod
    def from_namespace(
        cls,
        namespace: Any,
        *,
        cache_filename: str,
        project_root: str | os.PathLike[str] = PROJECT_ROOT,
        target_cache_path: str | None = None,
    ) -> "LayoutGenerationSettings":
        """Capture the current viewer namespace using the CLI's exact schema."""
        defaults = {
            "ALIGNMENT_SCORE": "global",
            "NORM_MODE": "alignment_length",
            "SIMILARITY_THRESHOLD": None,
            "TOP_EDGE_PERCENT": None,
            "UMAP_MODE": False,
            "UMAP_NEIGHBORS": 15,
            "UMAP_MIN_DIST": 0.1,
            "PHYSICS_ENGINE": "Molecular Dynamics (Style)",
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
            "SGLD_MIN_K": 20,
            "SGLD_K_PERCENT": 0.01,
            "SGLD_START_TEMP": 1.5,
            "SGLD_NOISE_SCALE": 1.0,
            "BOX_SCALE": 2.0,
            "PACKING_PADDING": 10.0,
            "MAX_FORCE_LIMIT": 20.0,
            "MAX_TOTAL_REPULSION_FORCE": 0.0,
        }
        values = {
            key: getattr(namespace, key, default)
            for key, default in defaults.items()
        }
        values.update(
            {
                "SAVED_LAYOUT_DIR": getattr(namespace, "SAVED_LAYOUT_DIR"),
                "NODE_FASTA_FILE": getattr(namespace, "NODE_FASTA_FILE"),
                "INPUT_HDF5": getattr(namespace, "INPUT_HDF5"),
                "CACHE_FILENAME": cache_filename,
            }
        )
        # EMAP-SSN Configuration historically stores several numeric widget values as strings.
        for key, value in tuple(values.items()):
            default = defaults.get(key)
            if isinstance(default, bool):
                if isinstance(value, str):
                    values[key] = value.strip().lower() in {"true", "1", "yes", "on"}
            elif isinstance(default, int) and isinstance(value, str):
                values[key] = int(value)
            elif isinstance(default, float) and isinstance(value, str):
                values[key] = float(value)
        for key in ("SIMILARITY_THRESHOLD", "TOP_EDGE_PERCENT"):
            if values[key] is None or str(values[key]).strip() in {"", "None"}:
                values[key] = None
            elif isinstance(values[key], str):
                values[key] = float(values[key])

        root = Path(project_root).resolve()
        for key in ("SAVED_LAYOUT_DIR", "NODE_FASTA_FILE", "INPUT_HDF5"):
            values[key] = _resolve_project_path(values[key], root)
        settings = cls(**values, _target_cache_path=target_cache_path)
        settings.validate()
        return settings

    def validate(self) -> None:
        cache_manifest.validate_cache_filename(self.CACHE_FILENAME)
        if not isinstance(self.UMAP_MODE, bool):
            raise LayoutGenerationError("UMAP_MODE must be a JSON boolean.")
        if not isinstance(self.ENABLE_PROGRESSIVE_SIMULATION, bool):
            raise LayoutGenerationError(
                "ENABLE_PROGRESSIVE_SIMULATION must be a JSON boolean."
            )
        if self.ALIGNMENT_SCORE not in {None, "global", "local"}:
            raise LayoutGenerationError("ALIGNMENT_SCORE must be global, local, or null.")
        if self.NORM_MODE not in {
            None,
            "alignment_length",
            "shorter_sequence",
            "longer_sequence",
            "average_sequence",
        }:
            raise LayoutGenerationError("NORM_MODE is not supported.")
        if self.ALIGNMENT_SCORE == "local" and self.NORM_MODE == "alignment_length":
            raise LayoutGenerationError(
                "NORM_MODE alignment_length is unavailable for local alignment scores."
            )
        if self.PHYSICS_ENGINE not in {
            "Molecular Dynamics (Style)",
            "Monte Carlo (Style)",
        }:
            raise LayoutGenerationError("PHYSICS_ENGINE is not supported.")
        if self.PACKING_GEOMETRY not in {"Square", "Circle"}:
            raise LayoutGenerationError("PACKING_GEOMETRY must be Square or Circle.")
        if not isinstance(self.LAYOUT_DEVICE_SELECTION, str) or not self.LAYOUT_DEVICE_SELECTION:
            raise LayoutGenerationError("LAYOUT_DEVICE_SELECTION must be a non-empty string.")

        optional_numbers = {
            "SIMILARITY_THRESHOLD": (self.SIMILARITY_THRESHOLD, None, None),
            "TOP_EDGE_PERCENT": (self.TOP_EDGE_PERCENT, 0.0, 100.0),
        }
        for name, (value, minimum, maximum) in optional_numbers.items():
            if value is not None:
                _finite_number(name, value, minimum=minimum, maximum=maximum)
        if not self.UMAP_MODE and self.TOP_EDGE_PERCENT is None and self.SIMILARITY_THRESHOLD is None:
            raise LayoutGenerationError(
                "Physics layout requires SIMILARITY_THRESHOLD or TOP_EDGE_PERCENT."
            )

        integer_ranges = {
            "UMAP_NEIGHBORS": (self.UMAP_NEIGHBORS, 2),
            "MAX_STEPS": (self.MAX_STEPS, 1),
            "RMSD_WINDOW": (self.RMSD_WINDOW, 1),
            "SGLD_MIN_K": (self.SGLD_MIN_K, 1),
        }
        for name, (value, minimum) in integer_ranges.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise LayoutGenerationError(
                    f"{name} must be an integer of at least {minimum}."
                )

        numeric_ranges = {
            "UMAP_MIN_DIST": (self.UMAP_MIN_DIST, 0.0, 1.0),
            "SPRING_K": (self.SPRING_K, 0.0, None),
            "COULOMB_K": (self.COULOMB_K, 0.0, None),
            "COULOMB_CUTOFF": (self.COULOMB_CUTOFF, 0.0, None),
            "DAMPING": (self.DAMPING, 0.0, None),
            "DT": (self.DT, 0.0, None),
            "RMSD_THRESHOLD": (self.RMSD_THRESHOLD, 0.0, None),
            "PERCENTAGE_DROP_THRESHOLD": (
                self.PERCENTAGE_DROP_THRESHOLD,
                0.0,
                None,
            ),
            "PACKING_GRID_SIZE": (self.PACKING_GRID_SIZE, 0.0, None),
            "SGLD_K_PERCENT": (self.SGLD_K_PERCENT, 0.0, None),
            "SGLD_START_TEMP": (self.SGLD_START_TEMP, 0.0, None),
            "SGLD_NOISE_SCALE": (self.SGLD_NOISE_SCALE, 0.0, None),
            "BOX_SCALE": (self.BOX_SCALE, 0.0, None),
            "PACKING_PADDING": (self.PACKING_PADDING, 0.0, None),
            "MAX_FORCE_LIMIT": (self.MAX_FORCE_LIMIT, 0.0, None),
            "MAX_TOTAL_REPULSION_FORCE": (
                self.MAX_TOTAL_REPULSION_FORCE,
                0.0,
                None,
            ),
        }
        for name, (value, minimum, maximum) in numeric_ranges.items():
            _finite_number(name, value, minimum=minimum, maximum=maximum)

    def to_document(
        self, *, project_root: str | os.PathLike[str] = PROJECT_ROOT
    ) -> dict[str, Any]:
        root = Path(project_root).resolve()
        values = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_") and item.name != "SAVED_LAYOUT_DIR"
        }
        values["NODE_FASTA_FILE"] = _portable_path(self.NODE_FASTA_FILE, root)
        values["INPUT_HDF5"] = _portable_path(self.INPUT_HDF5, root)
        return {
            "DIRECTORIES": {
                "SAVED_LAYOUT_DIR": _portable_path(self.SAVED_LAYOUT_DIR, root)
            },
            SCRIPT_NAME: values,
        }

    def engine_params(self) -> dict[str, Any]:
        excluded = {
            "SAVED_LAYOUT_DIR",
            "NODE_FASTA_FILE",
            "INPUT_HDF5",
            "CACHE_FILENAME",
            "ALIGNMENT_SCORE",
            "NORM_MODE",
            "TOP_EDGE_PERCENT",
        }
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_") and item.name not in excluded
        }


@dataclass
class LayoutGenerationResult:
    cache_path: str
    manifest: dict[str, Any]
    full_headers: list[str]
    positions: np.ndarray
    edges: np.ndarray
    edge_scores: np.ndarray
    box_limit: float
    fasta_records: list[tuple[str, str]]
    effective_similarity_threshold: float | None


def _manifest_settings(settings: LayoutGenerationSettings) -> dict[str, Any]:
    return {
        "alignment_score": settings.ALIGNMENT_SCORE,
        "normalization": settings.NORM_MODE,
        "umap_mode": settings.UMAP_MODE,
        "umap_neighbors": settings.UMAP_NEIGHBORS,
        "top_edge_percent": settings.TOP_EDGE_PERCENT,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
    }


def _resolve_cache_target(
    settings: LayoutGenerationSettings, manifest: Mapping[str, Any]
) -> tuple[str, str]:
    saved_root = os.path.abspath(os.path.normpath(settings.SAVED_LAYOUT_DIR))
    matches = cache_manifest.find_matching_manifest_folders(
        saved_root, manifest["compatibility"]
    )
    if len(matches) > 1:
        folders = ", ".join(item["folder"] for item in matches)
        raise LayoutGenerationError(
            f"Multiple compatible layout-cache folders were found: {folders}"
        )

    if settings._target_cache_path:
        requested = os.path.abspath(os.path.normpath(settings._target_cache_path))
        relative = cache_manifest.relative_cache_path(
            saved_root, os.path.dirname(requested), os.path.basename(requested)
        )
        requested = cache_manifest.resolve_relative_cache_path(saved_root, relative)
        if os.path.basename(requested) != settings.CACHE_FILENAME:
            raise LayoutGenerationError(
                "The viewer target filename does not match CACHE_FILENAME."
            )
        folder = os.path.dirname(requested)
        if matches and os.path.normcase(matches[0]["folder"]) != os.path.normcase(folder):
            raise LayoutGenerationError(
                "The viewer target folder differs from the compatible manifest folder."
            )
    elif matches:
        folder = matches[0]["folder"]
    else:
        network_type = manifest["compatibility"]["network_type"]
        folder_name = cache_manifest.build_canonical_cache_name(
            settings.NODE_FASTA_FILE,
            settings.INPUT_HDF5,
            network_type,
            **_manifest_settings(settings),
        )
        folder = cache_manifest.resolve_default_cache_folder(
            saved_root, folder_name, manifest["compatibility"]
        )

    manifest_path = os.path.join(folder, cache_manifest.MANIFEST_FILENAME)
    if os.path.isfile(manifest_path):
        try:
            existing = cache_manifest.read_manifest(folder)
        except Exception as error:
            raise LayoutGenerationError(
                f"The target folder contains an invalid cache manifest: {error}"
            ) from error
        if existing["manifest_id"] != manifest["manifest_id"]:
            raise LayoutGenerationError(
                "The target folder contains an incompatible cache manifest."
            )
    elif os.path.isdir(folder):
        allowed_backup = os.path.basename(settings.NODE_FASTA_FILE)
        unexpected = [
            entry.name
            for entry in os.scandir(folder)
            if entry.name != allowed_backup and not entry.name.endswith(".partial")
        ]
        if unexpected:
            raise LayoutGenerationError(
                "The canonical target folder already contains files but no compatible "
                "manifest: " + ", ".join(sorted(unexpected))
            )

    cache_path = os.path.join(folder, settings.CACHE_FILENAME)
    if os.path.exists(cache_path):
        raise FileExistsError(
            f"Layout cache already exists and will not be overwritten: {cache_path}"
        )
    return folder, cache_path


def _stage_json(folder: str, filename: str, payload: Mapping[str, Any]) -> str:
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".partial", dir=folder
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return staged


def _stage_fasta(folder: str, filename: str, records: list[tuple[str, str]]) -> str:
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".partial", dir=folder
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return staged


def _publish_auxiliary(staged: str, destination: str, expected_bytes: bytes) -> None:
    if os.path.exists(destination):
        with open(destination, "rb") as existing:
            if existing.read() != expected_bytes:
                raise LayoutGenerationError(
                    f"Existing cache artifact does not match this run: {destination}"
                )
        os.unlink(staged)
        return
    try:
        os.link(staged, destination)
    except FileExistsError:
        with open(destination, "rb") as existing:
            if existing.read() != expected_bytes:
                raise LayoutGenerationError(
                    f"Cache artifact was concurrently replaced: {destination}"
                )
    os.unlink(staged)


def _publish_fasta_backup(staged: str, destination: str) -> None:
    """Publish the canonical sanitized backup; replacing an older backup is safe."""
    os.replace(staged, destination)


def _publish_cache_without_overwrite(staged: str, destination: str) -> None:
    try:
        os.link(staged, destination)
    except FileExistsError as error:
        raise FileExistsError(
            f"Layout cache already exists and will not be overwritten: {destination}"
        ) from error
    os.unlink(staged)


def generate_layout_cache(
    settings: LayoutGenerationSettings,
) -> LayoutGenerationResult:
    """Generate and atomically publish one minimal layout cache."""
    settings.validate()
    if not os.path.isfile(settings.NODE_FASTA_FILE):
        raise FileNotFoundError(f"Input FASTA was not found: {settings.NODE_FASTA_FILE}")
    if not os.path.isfile(settings.INPUT_HDF5):
        raise FileNotFoundError(f"Input network was not found: {settings.INPUT_HDF5}")

    manifest = cache_manifest.build_manifest_for_files(
        settings.NODE_FASTA_FILE,
        settings.INPUT_HDF5,
        **_manifest_settings(settings),
    )
    if manifest["compatibility"]["network_type"] == "alignment" and (
        settings.ALIGNMENT_SCORE is None or settings.NORM_MODE is None
    ):
        raise LayoutGenerationError(
            "Alignment networks require ALIGNMENT_SCORE and NORM_MODE."
        )
    cache_folder, cache_path = _resolve_cache_target(settings, manifest)
    existing_manifest_path = os.path.join(
        cache_folder, cache_manifest.MANIFEST_FILENAME
    )
    if os.path.isfile(existing_manifest_path):
        manifest = cache_manifest.read_manifest(
            cache_folder, manifest["compatibility"]
        )

    headers, sequences, _ = load_sanitized_fasta(settings.NODE_FASTA_FILE)
    records = list(zip(headers, sequences))

    preparation_settings = SimpleNamespace(**{
        item.name: getattr(settings, item.name)
        for item in fields(settings)
        if not item.name.startswith("_")
    })
    with h5py.File(settings.INPUT_HDF5, "r") as raw_data:
        full_headers, edges, edge_scores = prepare_network(
            raw_data,
            settings=preparation_settings,
            selected_fasta_headers=headers,
        )

    n_nodes = len(full_headers)
    if n_nodes == 0:
        raise LayoutGenerationError(
            "No input FASTA sequences matched nodes in the selected network."
        )
    connectivity = (
        np.column_stack((edges, edge_scores))
        if len(edges)
        else np.zeros((0, 3), dtype=np.float32)
    )
    if settings.UMAP_MODE:
        import Layout_Engine_UMAP as layout_engine
    elif settings.PHYSICS_ENGINE == "Monte Carlo (Style)":
        import Layout_Engine_SSN_MonteCarlo as layout_engine
    else:
        import Layout_Engine_SSN_MolecularDynamics as layout_engine

    params = settings.engine_params()
    params["SIMILARITY_THRESHOLD"] = preparation_settings.SIMILARITY_THRESHOLD
    positions, box_limit = layout_engine.calculate_layout(
        connectivity, n_nodes, params
    )
    positions = np.asarray(positions, dtype=np.float32)
    if positions.shape != (n_nodes, 2) or not np.isfinite(positions).all():
        raise LayoutGenerationError(
            f"Layout engine returned invalid positions with shape {positions.shape}."
        )

    os.makedirs(cache_folder, exist_ok=True)
    backup_name = os.path.basename(settings.NODE_FASTA_FILE)
    if backup_name in {settings.CACHE_FILENAME, cache_manifest.MANIFEST_FILENAME}:
        raise LayoutGenerationError("The FASTA backup name conflicts with a cache artifact.")
    backup_path = os.path.join(cache_folder, backup_name)
    manifest_path = os.path.join(cache_folder, cache_manifest.MANIFEST_FILENAME)

    staged_paths: list[str] = []
    try:
        staged_fasta = _stage_fasta(cache_folder, backup_name, records)
        staged_paths.append(staged_fasta)
        staged_manifest = None
        if not os.path.exists(manifest_path):
            staged_manifest = _stage_json(
                cache_folder, cache_manifest.MANIFEST_FILENAME, manifest
            )
            staged_paths.append(staged_manifest)

        descriptor, staged_cache = tempfile.mkstemp(
            prefix=f".{settings.CACHE_FILENAME}.",
            suffix=".partial",
            dir=cache_folder,
        )
        os.close(descriptor)
        staged_paths.append(staged_cache)
        with h5py.File(staged_cache, "w") as output:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            output.attrs["cache_manifest_id"] = manifest["manifest_id"]
            output.create_dataset(
                "headers",
                data=np.asarray(full_headers, dtype=object),
                dtype=string_dtype,
                compression="gzip",
            )
            output.create_dataset("positions", data=positions, compression="gzip")
            output.flush()

        _publish_fasta_backup(staged_fasta, backup_path)
        staged_paths.remove(staged_fasta)
        if staged_manifest is not None:
            with open(staged_manifest, "rb") as handle:
                manifest_bytes = handle.read()
            _publish_auxiliary(staged_manifest, manifest_path, manifest_bytes)
            staged_paths.remove(staged_manifest)
        _publish_cache_without_overwrite(staged_cache, cache_path)
        staged_paths.remove(staged_cache)
    finally:
        for staged in staged_paths:
            try:
                os.unlink(staged)
            except FileNotFoundError:
                pass

    print(f"Layout saved to: {cache_path}")
    return LayoutGenerationResult(
        cache_path=cache_path,
        manifest=dict(manifest),
        full_headers=list(full_headers),
        positions=positions,
        edges=np.asarray(edges, dtype=np.int32),
        edge_scores=np.asarray(edge_scores, dtype=np.float32),
        box_limit=float(box_limit),
        fasta_records=records,
        effective_similarity_threshold=preparation_settings.SIMILARITY_THRESHOLD,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an SSN layout cache without opening EMAP-SSN Viewer."
    )
    parser.add_argument(
        "settings_json",
        help="Layout settings JSON exported by emapssn_config.py",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        settings = LayoutGenerationSettings.from_json_file(args.settings_json)
        result = generate_layout_cache(settings)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(result.cache_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
