# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Settings export shared by GUI controls, headless entrypoints and MCP.

Personal settings are read only during export. Execution consumes explicit JSON.
GUI imports must stay out of this module.
"""
from copy import deepcopy
import json
import ntpath
import os
from pathlib import Path
import posixpath
from types import SimpleNamespace
import uuid


def read_object(path, *, optional=False):
    path = Path(path)
    if optional and not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read settings '{path}': {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"Settings '{path}' must contain a JSON object.")
    return document


def absolute_path(value, root):
    return os.path.abspath(os.path.join(root, os.path.expandvars(os.path.expanduser(os.fspath(value)))))


def build_pipeline_export(tool_id, directories, values, project_root, *, absolute=False):
    from mcp_server.Pipeline_Settings import serialize_export_settings
    from utilities.Tool_Execution import build_settings_document, get_tool_spec
    spec = get_tool_spec(tool_id)
    values = serialize_export_settings(tool_id, values, project_root)
    directories = {key: directories.get(key, "") for key in spec.required_directories}
    if absolute:
        directories = {key: absolute_path(value, project_root) for key, value in directories.items()}
        values = {key: absolute_path(value, project_root)
                  if key.endswith("_DIR") and isinstance(value, str) and value.strip() else value
                  for key, value in values.items()}
    else:
        # Preserve the GUI's portable relative-directory representation.
        def portable(value):
            if not value:
                return ""
            value = os.fspath(value)
            return value if ntpath.isabs(value) or posixpath.isabs(value) else value.replace("\\", "/")
        directories = {key: portable(value) for key, value in directories.items()}
        values = {key: portable(value) if key.endswith("_DIR") else value for key, value in values.items()}
    return build_settings_document(spec, directories, values)


def write_export(document, project_root, export_directory, stem, output_path=None):
    """Publish a complete export exclusively; never overwrite another export."""
    directory = Path(absolute_path(export_directory, project_root))
    target = Path(absolute_path(output_path, project_root)) if output_path else directory / f"{stem}-{uuid.uuid4().hex}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + f".{uuid.uuid4().hex}.partial")
    try:
        staged.write_text(json.dumps(document, indent=4, allow_nan=False) + "\n", encoding="utf-8")
        os.link(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    return {"settings_path": str(target), "settings_document": document}


def export_pipeline_settings(tool_id, project_root, output_path=None):
    from utilities.Tool_Directories import DEFAULT_DIRECTORY_PATHS
    from utilities.Tool_Execution import get_tool_spec
    from mcp_server.Pipeline_Settings import get_pipeline_schema
    spec = get_tool_spec(tool_id)
    saved = read_object(Path(project_root) / "tools_settings.json", optional=True)
    directories = saved.get("DIRECTORIES", {})
    values = saved.get(spec.settings_section, {})
    if not isinstance(directories, dict) or not isinstance(values, dict):
        raise ValueError("Saved directories and tool settings must be JSON objects.")
    resolved_directories = {}
    for key, default in DEFAULT_DIRECTORY_PATHS.items():
        value = directories.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"DIRECTORIES.{key} must be a string or null.")
        resolved_directories[key] = value if value and value.strip() else default
    directories = resolved_directories
    schema = get_pipeline_schema(tool_id, project_root)
    defaults = {key: deepcopy(prop["default"]) for key, prop in schema["parameters_schema"]["properties"].items()}
    for key, prop in schema["parameters_schema"]["properties"].items():
        if prop.get("x-choice-provider") and defaults[key] not in prop.get("enum", ()):
            defaults[key] = None if prop["default"] is None else ""
    defaults.update(values)
    document = build_pipeline_export(tool_id, directories, defaults, project_root, absolute=True)
    return write_export(document, project_root, directories["SETTING_EXPORT_DIR"], tool_id, output_path)


def build_layout_export(values, project_root, *, cache_filename=None, target_cache_path=None,
                        automatic=False, selection_resolved=False):
    """Serialize generation fields; GUI may supply its already-discovered selection."""
    from Layout_Cache_Generator import LayoutGenerationSettings, resolve_layout_selection
    settings = LayoutGenerationSettings.from_namespace(
        SimpleNamespace(**values), cache_filename=cache_filename or "version_00.h5",
        project_root=project_root, target_cache_path=target_cache_path,
    )
    settings.CACHE_NAME_MODE = "auto" if automatic or not cache_filename else "explicit"
    settings.TARGET_CACHE_PATH = target_cache_path
    if not selection_resolved:
        resolve_layout_selection(settings)
    document = settings.to_document(project_root=project_root)
    document["DIRECTORIES"]["SAVED_LAYOUT_DIR"] = settings.SAVED_LAYOUT_DIR
    section = document["Layout_Cache_Generator.py"]
    section.update(NODE_FASTA_FILE=settings.NODE_FASTA_FILE, INPUT_HDF5=settings.INPUT_HDF5)
    return document


def _config_export_directory(values, project_root):
    from utilities.Viewer_Settings import ALIASES
    def resolve(value, seen=()):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Config directories must be nonempty strings.")
        for alias, key in ALIASES.items():
            if value == alias or value.startswith((alias + "/", alias + "\\")):
                if key in seen:
                    raise ValueError(f"Circular directory alias: {key}")
                value = os.path.join(resolve(values[key], (*seen, key)), value[len(alias):].lstrip("/\\"))
                break
        return absolute_path(value, project_root)
    return resolve(values["SETTING_EXPORT_DIR"])


def config_export_document(kind, project_root, *, settings_path=None):
    from utilities.Viewer_Settings import DEFAULTS, normalize_viewer_settings, validate_viewer_document
    from utilities.Viewer_Defaults import DIRECTORY_PROFILE_DEFAULTS, LEGACY_DEFAULT_DIRECTORY_PATHS
    from Layout_Cache_Generator import LayoutGenerationSettings, resolve_layout_selection
    from Cache_Manifest import find_matching_manifest_folders
    if kind not in {"layout", "viewer"}:
        raise ValueError("kind must be layout or viewer.")
    saved = read_object(Path(project_root) / "viewer_settings.json", optional=True)
    # Saved preferences can contain historical GUI-only keys; explicit overlays cannot.
    values = deepcopy(DEFAULTS)
    values.update({key: value for key, value in saved.items() if key in DEFAULTS})
    if str(values["SAVED_CONFIG_DIR"]).replace("\\", "/").rstrip("/").casefold() in {"saved_config", "cache_files/saved_config"}:
        values["SAVED_CONFIG_DIR"] = DEFAULTS["SAVED_CONFIG_DIR"]
    for key, legacy in LEGACY_DEFAULT_DIRECTORY_PATHS.items():
        if os.path.normpath(str(values[key])) == os.path.normpath(legacy):
            values[key] = DIRECTORY_PROFILE_DEFAULTS[key]
    overlay = read_object(absolute_path(settings_path, project_root)) if settings_path else {}
    layout_overlay = overlay.get("Layout_Cache_Generator.py")
    if layout_overlay is not None:
        # Re-export an edited layout document using its explicit values, not GUI defaults.
        settings = LayoutGenerationSettings.from_document(overlay, project_root=project_root)
        resolve_layout_selection(settings)
        if kind != "layout":
            raise ValueError("Viewer export requires a Viewer settings overlay with the generated cache path.")
        document = settings.to_document(project_root=project_root)
        document["DIRECTORIES"]["SAVED_LAYOUT_DIR"] = settings.SAVED_LAYOUT_DIR
        document["Layout_Cache_Generator.py"].update(NODE_FASTA_FILE=settings.NODE_FASTA_FILE, INPUT_HDF5=settings.INPUT_HDF5)
        return document, _config_export_directory(values, project_root)
    unknown = set(overlay) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown Config settings: " + ", ".join(sorted(unknown)))
    values.update(overlay)
    # Match Config's export/launch handling of inactive filter controls.
    umap = str(values["UMAP_MODE"]).strip().lower() in {"true", "1"}
    top_active = values["TOP_EDGE_PERCENT"] is not None and str(values["TOP_EDGE_PERCENT"]).strip().lower() not in {"", "none"}
    if umap:
        values["TOP_EDGE_PERCENT"] = None
        values["SIMILARITY_THRESHOLD"] = None
    elif top_active:
        values["SIMILARITY_THRESHOLD"] = None
    selected = values.get("TARGET_CACHE_PATH") or None
    selected_filename = values.get("CACHE_FILENAME", "")
    values["TARGET_CACHE_PATH"] = selected or ""
    if not selected:
        values["CACHE_FILENAME"] = ""
    values = normalize_viewer_settings(values, project_root, require_cache=False)
    if kind == "layout":
        document = build_layout_export(values, project_root)
    else:
        if not selected:
            settings = LayoutGenerationSettings.from_namespace(SimpleNamespace(**values), cache_filename="version_00.h5", project_root=project_root)
            settings.CACHE_NAME_MODE = "auto"
            manifest = resolve_layout_selection(settings)
            matches = find_matching_manifest_folders(settings.SAVED_LAYOUT_DIR, manifest["compatibility"])
            candidates = [] if not matches else [p for p in Path(matches[0]["folder"]).iterdir() if p.is_file() and p.suffix.lower() == ".h5"]
            if not candidates:
                raise ValueError("No compatible layout cache exists; generate a layout first.")
            chosen = sorted(candidates, key=lambda p: (-p.stat().st_mtime_ns, p.name))[0]
            values["TARGET_CACHE_PATH"] = str(chosen)
            values["CACHE_FILENAME"] = chosen.name
        else:
            values["CACHE_FILENAME"] = selected_filename or Path(values["TARGET_CACHE_PATH"]).name
        document = validate_viewer_document(values, project_root)
    return document, values["SETTING_EXPORT_DIR"]


def export_config_settings(kind, project_root, output_path=None, settings_path=None):
    document, directory = config_export_document(kind, project_root, settings_path=settings_path)
    result = write_export(document, project_root, directory, f"{kind}-settings", output_path)
    section = document["Layout_Cache_Generator.py"] if kind == "layout" else document
    result.update(cache_path=section["TARGET_CACHE_PATH"], cache_filename=section["CACHE_FILENAME"])
    return result


def main(application, argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=f"Headless {application} settings and execution")
    parser.add_argument("--headless", action="store_true", required=True)
    commands = parser.add_subparsers(dest="operation", required=True)
    export = commands.add_parser("export")
    export.add_argument("--output")
    if application == "tools":
        export.add_argument("--tool", required=True)
    else:
        export.add_argument("--kind", choices=("layout", "viewer"), required=True)
        export.add_argument("--settings", help="Edited settings overlay for re-export")
        generate = commands.add_parser("generate-layout")
        generate.add_argument("--settings", required=True)
        generate.add_argument("--result", help="Private machine-readable job result")
        launch = commands.add_parser("launch-viewer")
        launch.add_argument("--settings", required=True)
        launch.add_argument("--viewer-mode", choices=("normal", "headless"), default="normal")
        launch.add_argument("--delete-settings", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        if args.operation == "export":
            result = export_pipeline_settings(args.tool, root, args.output) if application == "tools" else export_config_settings(args.kind, root, args.output, args.settings)
            print(json.dumps(result, allow_nan=False))
        elif args.operation == "generate-layout":
            from Layout_Cache_Generator import LayoutGenerationSettings, generate_layout_cache
            from utilities.Tool_Execution import write_json_document
            settings = LayoutGenerationSettings.from_json_file(absolute_path(args.settings, root), project_root=root)
            generated = generate_layout_cache(settings)
            result = {"TARGET_CACHE_PATH": generated.cache_path, "CACHE_FILENAME": Path(generated.cache_path).name,
                      "SAVED_LAYOUT_DIR": settings.SAVED_LAYOUT_DIR}
            if args.result:
                write_json_document(args.result, result)
                write_json_document(str(args.result) + ".settings.json", settings.to_document(project_root=root))
            print(json.dumps(result))
        else:
            # In-process dispatch preserves the detached launcher's PID on Windows too.
            # Viewer owns validation, explicit-settings isolation and snapshot deletion.
            import runpy
            script = root / "src" / "EMAPSSN_Viewer.py"
            sys.argv = [str(script), "--settings", absolute_path(args.settings, root)]
            if args.delete_settings:
                sys.argv.append("--delete-settings")
            if args.viewer_mode == "headless":
                sys.argv.append("--headless")
            else:
                os.environ.pop("SSN_VIEWER_HEADLESS", None)
                if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
                    os.environ.pop("QT_QPA_PLATFORM", None)
            runpy.run_path(str(script), run_name="__main__")
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
