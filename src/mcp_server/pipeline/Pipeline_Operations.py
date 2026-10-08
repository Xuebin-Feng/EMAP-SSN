# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Pipeline and layout-cache operation handlers for MCP workflows."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
from typing import Annotated, Any, Literal

_CORE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _CORE_DIR.parents[1]
_PROJECT_ROOT = str(_CORE_DIR.parents[2])

for _path_str in (str(_SRC_DIR), _PROJECT_ROOT):
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from mcp_server.core.App_Context import AppContext, _context
from mcp_server.pipeline.Pipeline_Guide import path_rules, tool_guide, workflows
from mcp_server.pipeline.Pipeline_Jobs import MAX_WAIT_SECONDS, PipelineJobError
from mcp_server.pipeline.Pipeline_Settings import (
    DESCRIPTIONS,
    PipelineSettingsError,
    get_pipeline_schema,
    normalize_pipeline_settings,
)
from tools.tool_helpers.Tool_Pipeline import list_tool_specs

_LAYOUT_CONTRACT = json.loads(
    Path(__file__).with_name("Layout_Settings_Contract.json").read_text(encoding="utf-8")
)


def layout_defaults() -> dict[str, Any]:
    """Built-in layout values that start_layout_job's individual arguments fall back to.

    Tests pin these to start_layout_job's literal payload and to the engine.
    """
    return {key: deepcopy(field["default"]) for key, field in _LAYOUT_CONTRACT["fields"].items()
            if "default" in field}


class PipelineToolInfo(BaseModel):
    tool_id: str
    description: str
    script_name: str
    settings_section: str
    required_directories: list[str]
    output_directories: list[str]
    stage: str
    purpose: str
    inputs: dict[str, dict[str, Any]]
    outputs: list[dict[str, Any]]
    next: list[str]
    requirements: str | None
    notes: list[str]


class PipelineCatalog(BaseModel):
    tools: list[PipelineToolInfo]
    layout: dict[str, Any]
    workflows: list[dict[str, Any]]
    path_rules: list[str]
    max_running: int
    max_pending: int


class OutputFile(BaseModel):
    path: str
    change: Literal["created", "modified", "deleted"]
    size_bytes: int | None


class PipelineJobInfo(BaseModel):
    job_id: str
    tool_id: str
    status: Literal[
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelling",
        "cancelled",
    ]
    queue_position: int | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    failure_message: str | None
    cancellation_requested: bool
    settings_snapshot: str
    stdout_log: str
    stderr_log: str
    output_locations: dict[str, str]
    output_files: list[OutputFile] = Field(default_factory=list, description=(
        "Files the job created, modified or deleted in the folders it writes; empty while it runs."))
    output_files_omitted: int = 0
    result: dict[str, Any] | None = Field(default=None, description=(
        "Layout jobs: node, edge, cluster and threshold summary of the published cache."))
    latest_output: dict[str, str | None] | None = Field(default=None, description=(
        "Last line of stdout and stderr (progress), from get_job and wait_job."))


class PipelineJobList(BaseModel):
    jobs: list[PipelineJobInfo]


class PipelineLogPage(BaseModel):
    job_id: str
    stream: Literal["stdout", "stderr"]
    offset: int
    next_offset: int
    size: int
    eof: bool
    text: str


def _job_info(payload: dict[str, Any]) -> PipelineJobInfo:
    return PipelineJobInfo.model_validate(payload)


# Layout fields that start_layout_job's `parameters` must not carry: each has
# its own argument, or is fixed because a layout job always publishes a new cache.
_LAYOUT_RESERVED_KEYS = {
    "NODE_FASTA_FILE": "the node_fasta_file argument",
    "INPUT_HDF5": "the input_hdf5 argument",
    "CACHE_FILENAME": "the cache_filename argument",
    "CACHE_NAME_MODE": "the cache_filename argument",
    "SAVED_LAYOUT_DIR": "directories={'SAVED_LAYOUT_DIR': ...}",
    "TARGET_CACHE_PATH": None,
}


# export_layout_settings takes the inputs as arguments; cache naming and the
# layout root are ordinary export fields there.
_EXPORT_RESERVED_KEYS = {
    "NODE_FASTA_FILE": "the node_fasta_file argument",
    "INPUT_HDF5": "the input_hdf5 argument",
    "TARGET_CACHE_PATH": None,
}


def _layout_overrides(parameters: Any, reserved: dict[str, str | None] = _LAYOUT_RESERVED_KEYS) -> dict[str, Any]:
    """Validate a layout `parameters` object and return it upper-cased.

    Keys match the layout document's fields case-insensitively. Anything else
    is rejected, with the closest field name as a hint, instead of being
    silently dropped when the document is encoded.
    """
    from difflib import get_close_matches
    from desktop.Viewer_State import sections

    if not isinstance(parameters, dict):
        raise ToolError("parameters must be an object of layout fields.")
    allowed = {key for keys in sections("layout").values() for key in keys}
    allowed -= set(reserved)
    overrides: dict[str, Any] = {}
    problems: list[str] = []
    for key, value in parameters.items():
        name = str(key).upper()
        if name in overrides:
            problems.append(f"'{key}' is given twice")
        elif name in reserved:
            use = reserved[name]
            problems.append(
                f"'{key}' is not a layout override; "
                + (f"use {use}" if use else "a layout job always publishes a new cache")
            )
        elif name not in allowed:
            match = get_close_matches(name, sorted(allowed), n=1, cutoff=0.75)
            problems.append(
                f"'{key}' is not a layout setting"
                + (f" (did you mean {match[0]}?)" if match else "")
            )
        else:
            overrides[name] = value
    if problems:
        raise ToolError(
            "Invalid parameters: " + "; ".join(problems)
            + ". Accepted keys: " + ", ".join(sorted(allowed)) + "."
        )
    return overrides


def list_pipeline_tools() -> PipelineCatalog:
    """Plan which pipelines to run and in what order. Returns each tool_id with
    its stage, inputs (and the directory a relative name resolves in), output
    file names, overwrite behavior and typical next steps, plus workflow recipes
    and path rules. These IDs are arguments, not MCP tool names. Next call
    get_pipeline_tool_schema with the chosen tool_id. Layout calculation uses
    start_layout_job separately; its jobs report tool_id generate_layout_cache.
    """
    return PipelineCatalog(
        tools=[
            PipelineToolInfo(
                tool_id=spec.tool_id,
                description=DESCRIPTIONS[spec.tool_id],
                script_name=spec.script_name,
                settings_section=spec.settings_section,
                required_directories=list(spec.required_directories),
                output_directories=list(spec.output_directories),
                **tool_guide(spec.tool_id),
            )
            for spec in list_tool_specs()
        ],
        layout={
            "actions": ["network_statistics", "get_layout_schema", "export_layout_settings", "start_layout_job"],
            "job_tool_id": "generate_layout_cache",
            "inputs": "A node FASTA and a network file from any network-building tool.",
            "outputs": "A new cache <SAVED_LAYOUT_DIR>/<inputs and filter>/version_NN.h5 with a cache_manifest.json "
                       "and a FASTA backup; existing caches are never overwritten.",
            "next": "Viewer settings (emapssn_viewer_control export_settings) to open the cache.",
        },
        workflows=workflows(),
        path_rules=path_rules(),
        max_running=1,
        max_pending=16,
    )


async def get_compute_capabilities(tool_id: str | None = None) -> dict[str, Any]:
    """Check available computation before choosing device settings for a job.
    Optional tool_id is a pipeline ID from list_pipeline_tools and adds applicable
    settings. Returns runtime devices and memory without benchmarks. Then prepare
    settings using get_pipeline_tool_schema. Metadata support is
    unverified by computation; physical devices unavailable to this runtime are omitted.
    """
    from mcp_server.pipeline.Compute_Capabilities import discover_compute_capabilities
    try:
        return await asyncio.to_thread(discover_compute_capabilities, _PROJECT_ROOT, tool_id)
    except KeyError as error:
        raise ToolError(str(error)) from error


async def inspect_pipeline_file(
    path: str,
    file_type: Literal["auto", "fasta", "alignment_fasta", "embedding", "network", "sparse_msa", "blast_tabular", "settings", "layout_cache"] = "auto",
    tool_id: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check a known input before execution, or an output after job success.
    Inspect a selected file read-only without scanning numerical HDF5 payloads.
    A FASTA report gives raw length quantiles, duplicate sequences, header
    annotation terms (partial, fragment, ... by exact spelling) and example
    headers; a network report gives its type, model, pair coverage and gap
    penalties, precision or BLAST matrix; an embedding report gives the model
    and feature_dimension. Relative paths use the project root (not a tool's
    input directory). Plain BLAST text may require file_type.
    Optional tool_id/parameters supplies inspection context (e.g. BLAST columns).
    Structure, generation completion, and pair coverage are separate conclusions;
    valid structure is not proof of numerical correctness or job readiness.
    Use the findings to prepare settings or qualify the reported result; this
    tool does not search directories or repair files.
    """
    from mcp_server.pipeline.Pipeline_File_Inspection import inspect_pipeline_file as inspect_file
    try:
        return await asyncio.to_thread(inspect_file, path, _PROJECT_ROOT, file_type, tool_id, parameters)
    except (KeyError, TypeError, ValueError) as error:
        raise ToolError(str(error)) from error


def get_pipeline_tool_schema(tool_id: str) -> dict[str, Any]:
    """Prepare a pipeline request after choosing tool_id from list_pipeline_tools.
    Returns accepted parameters, defaults, choices, conditions, directory rules,
    and an example. Next call validate_pipeline_settings with the intended values.
    This schema is not the layout or Viewer settings contract.
    """
    try:
        return get_pipeline_schema(tool_id, _PROJECT_ROOT)
    except (KeyError, OSError, ValueError) as error:
        raise ToolError(str(error)) from error


def validate_pipeline_settings(
    tool_id: str,
    parameters: dict[str, Any] | None = None,
    directories: dict[str, Any] | None = None,
    settings_document: dict[str, Any] | None = None,
    settings_path: str | None = None,
) -> dict[str, Any]:
    """Check pipeline settings before start_pipeline_job without creating files or jobs.
    Use tool_id from list_pipeline_tools. Supply exactly one
    of parameters, settings_document, or settings_path; directories accompanies
    parameters only. Numbers and booleans require native JSON types. This checks
    configuration, not file existence, model credentials, or hardware readiness.
    Inspect valid and field-specific errors; when valid, pass the returned
    settings_document to start_pipeline_job with the same tool_id.
    """
    return normalize_pipeline_settings(
        tool_id, _PROJECT_ROOT, parameters=parameters, directories=directories,
        settings_document=settings_document, settings_path=settings_path,
    )


async def start_pipeline_job(
    tool_id: Annotated[str, Field(description="Stable ID from list_pipeline_tools")],
    ctx: Context[AppContext],
    settings_document: Annotated[
        dict[str, Any] | None,
        Field(description="Complete exported or validated SSN settings document; use instead of parameters or settings_path"),
    ] = None,
    settings_path: Annotated[
        str | None,
        Field(description="Path to existing SSN settings JSON; use instead of parameters or settings_document"),
    ] = None,
    parameters: Annotated[
        dict[str, Any] | None,
        Field(description="Tool parameters from get_pipeline_tool_schema, using native JSON types"),
    ] = None,
    directories: Annotated[
        dict[str, Any] | None,
        Field(description="Optional directory overrides, only with parameters"),
    ] = None,
) -> PipelineJobInfo:
    """Run a chosen pipeline after schema discovery and settings preview.
    Validate and enqueue using exactly one of parameters,
    settings_document, or settings_path. Optional directories goes with parameters.
    Use validate_pipeline_settings for a preview; invalid requests never queue.
    Returns job_id and current status, not completed outputs. Next use
    wait_pipeline_job (or get_pipeline_job) and read_pipeline_log. Files may be
    created or overwritten according to settings (get_pipeline_tool_schema lists
    each output and what happens to an existing one); backend exit cancels this
    server's jobs.
    """
    preview = normalize_pipeline_settings(
        tool_id, _PROJECT_ROOT, parameters=parameters, directories=directories,
        settings_document=settings_document, settings_path=settings_path,
    )
    if not preview["valid"]:
        raise ToolError(str(PipelineSettingsError(preview["errors"])))
    try:
        payload = await _context(ctx).jobs.submit(tool_id, preview["settings_document"])
    except (KeyError, OSError, TypeError, ValueError, PipelineJobError) as error:
        raise ToolError(str(error)) from error
    return _job_info(payload)


async def start_layout_job(
    ctx: Context[AppContext],
    node_fasta_file: Annotated[
        str | None,
        Field(description="Node FASTA sequence file name or project-relative/absolute path"),
    ] = None,
    input_hdf5: Annotated[
        str | None,
        Field(description="Network similarity/distance HDF5 file name or project-relative/absolute path"),
    ] = None,
    cache_filename: Annotated[
        str | None,
        Field(description="Explicit target filename; omit to allocate the next free version automatically"),
    ] = None,
    similarity_threshold: Annotated[
        float | None,
        Field(description="Cutoff similarity threshold (either threshold or top_edge_percent required for physics)"),
    ] = None,
    top_edge_percent: Annotated[
        float | None,
        Field(description="Top edge percentage cutoff between 0.0 and 100.0"),
    ] = None,
    umap_mode: Annotated[
        bool,
        Field(description="Whether to use UMAP dimension reduction instead of physics simulation"),
    ] = False,
    layout_device_selection: Annotated[
        str,
        Field(description="Target compute device: 'auto', 'cpu', 'cuda:N', 'xpu:N', or 'mps'"),
    ] = "auto",
    alignment_score: Annotated[
        Literal["global", "local"] | None,
        Field(description="Alignment score mode: 'global' or 'local' (for alignment networks)"),
    ] = "global",
    norm_mode: Annotated[
        Literal["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"] | None,
        Field(description="Score normalization mode (for alignment networks)"),
    ] = "alignment_length",
    parameters: Annotated[
        dict[str, Any] | None,
        Field(description="Advanced layout overrides with individual inputs (e.g. SPRING_K, COULOMB_K, MAX_STEPS); not a pipeline parameters document"),
    ] = None,
    directories: Annotated[
        dict[str, Any] | None,
        Field(description="Optional directory overrides, e.g. {'SAVED_LAYOUT_DIR': '...'}"),
    ] = None,
    settings_document: Annotated[
        dict[str, Any] | None,
        Field(description="Layout generation document, for example from export_config_settings(kind='layout'); replaces individual inputs"),
    ] = None,
    settings_path: Annotated[
        str | None,
        Field(description="Path to an existing exported layout settings JSON file"),
    ] = None,
) -> PipelineJobInfo:
    """Calculate a layout using full JSON from export_config_settings(kind='layout').
    Choose the edge filter with network_statistics first. Export with
    node_fasta_file and input_hdf5 to inherit saved simulation and physics
    settings, change only requested fields and required dependencies, then pass
    the exported settings_document here. Individual arguments replace omitted
    preferences with built-in defaults (see get_layout_schema).
    Validate and enqueue into the shared pipeline FIFO queue; this is separate
    from the pipeline IDs in list_pipeline_tools and has no standalone validator.
    Calculates node coordinates via iterative force-directed physics or UMAP dimension
    reduction and publishes an HDF5 layout cache file along with a canonical FASTA backup and
    manifest. Coordinates are 2D unless LAYOUT_DIMENSIONS is set to 3 in parameters or the
    settings document; LAYOUT_SEED (default 42, or null for an unseeded run) seeds the layout.
    Supply either individual parameters, settings_document, or settings_path.
    With individual arguments, omitted settings use built-in defaults, SAVED_LAYOUT_DIR
    comes from the saved viewer_settings.json, and parameters keys must be layout fields.
    Follow the returned job_id with wait_pipeline_job; its result reports nodes,
    edges, the effective threshold, clusters and isolated nodes, and
    output_locations.TARGET_CACHE_PATH the cache. This does not launch a Viewer.
    """
    from Layout_Cache_Generator import LayoutGenerationSettings, LayoutGenerationError

    target_doc = None
    if settings_path is not None:
        if settings_document is not None or node_fasta_file is not None or input_hdf5 is not None:
            raise ToolError("Provide exactly one of individual parameters, settings_document, or settings_path.")
        path = os.fspath(settings_path)
        if not os.path.isabs(path):
            path = os.path.join(_PROJECT_ROOT, path)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                target_doc = json.load(handle)
        except Exception as error:
            raise ToolError(f"Could not read settings_path '{settings_path}': {error}") from error
    elif settings_document is not None:
        if node_fasta_file is not None or input_hdf5 is not None:
            raise ToolError("Provide exactly one of individual parameters, settings_document, or settings_path.")
        target_doc = dict(settings_document)
    else:
        if not node_fasta_file or not str(node_fasta_file).strip():
            raise ToolError("node_fasta_file is required when settings_document or settings_path is not supplied.")
        if not input_hdf5 or not str(input_hdf5).strip():
            raise ToolError("input_hdf5 is required when settings_document or settings_path is not supplied.")

        dirs = dict(directories) if isinstance(directories, dict) else {}
        saved_layout_dir = dirs.get("SAVED_LAYOUT_DIR")
        if saved_layout_dir:
            if not os.path.isabs(saved_layout_dir):
                saved_layout_dir = os.path.join(_PROJECT_ROOT, saved_layout_dir)
        else:
            from utilities.Headless_Settings import resolve_saved_directory
            try:
                saved_layout_dir = resolve_saved_directory("SAVED_LAYOUT_DIR", _PROJECT_ROOT)
            except ValueError as error:
                raise ToolError(str(error)) from error

        # Literal defaults: tests compare every hard-coded layout default in the
        # source tree, and get_layout_schema's contract must match these.
        payload: dict[str, Any] = {
            "NODE_FASTA_FILE": str(node_fasta_file).strip(),
            "INPUT_HDF5": str(input_hdf5).strip(),
            "CACHE_FILENAME": str(cache_filename).strip() if cache_filename is not None else "version_00.h5",
            "CACHE_NAME_MODE": "explicit" if cache_filename is not None else "auto",
            "ALIGNMENT_SCORE": alignment_score,
            "NORM_MODE": norm_mode,
            "SIMILARITY_THRESHOLD": similarity_threshold,
            "TOP_EDGE_PERCENT": top_edge_percent,
            "UMAP_MODE": bool(umap_mode),
            "UMAP_NEIGHBORS": 15,
            "UMAP_MIN_DIST": 0.1,
            "LAYOUT_DEVICE_SELECTION": layout_device_selection or "auto",
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
            "BOX_SCALE": 2.0,
            "PACKING_PADDING": 10.0,
            "MAX_FORCE_LIMIT": 20.0,
            "MAX_TOTAL_REPULSION_FORCE": 0.0,
            "LAYOUT_DIMENSIONS": 2,
            "LAYOUT_SEED": 42,
        }
        if parameters:
            payload.update(_layout_overrides(parameters))

        from desktop.Viewer_State import encode_document
        target_doc = encode_document("layout", {**payload, "SAVED_LAYOUT_DIR": saved_layout_dir, "TARGET_CACHE_PATH": None})

    try:
        settings = LayoutGenerationSettings.from_document(target_doc, project_root=_PROJECT_ROOT)
        payload = await _context(ctx).jobs.submit_layout_job(settings)
    except (LayoutGenerationError, ValueError, TypeError, KeyError, OSError, PipelineJobError) as error:
        raise ToolError(str(error)) from error

    return _job_info(payload)


async def list_pipeline_jobs(
    ctx: Context[AppContext],
    limit: Annotated[int, Field(ge=1, le=100, description="Maximum number of newest server-owned jobs to return")] = 100,
) -> PipelineJobList:
    """Find recent pipeline or layout job IDs owned by this STDIO server.
    limit bounds the newest jobs returned. Next use get_pipeline_job for an ID
    and read_pipeline_log for progress or failures. History is server-local.
    """
    try:
        jobs = await _context(ctx).jobs.list_jobs(limit=limit)
    except PipelineJobError as error:
        raise ToolError(str(error)) from error
    return PipelineJobList(jobs=[_job_info(job) for job in jobs])


async def get_pipeline_job(
    job_id: str,
    ctx: Context[AppContext],
) -> PipelineJobInfo:
    """Follow a job_id returned by either start tool or list_pipeline_jobs.
    Returns status, failure_message, output locations and, once finished, the
    output_files the job created, modified or deleted (none means nothing was
    written, for example because an existing result was kept). Queued/running is
    not success; prefer wait_pipeline_job to repeated calls. On failure use
    read_pipeline_log for both streams; after succeeded use inspect_pipeline_file
    on relevant output files.
    """
    try:
        return _job_info(await _context(ctx).jobs.get_job(job_id))
    except PipelineJobError as error:
        raise ToolError(str(error)) from error


async def wait_pipeline_job(
    job_id: str,
    ctx: Context[AppContext],
    timeout_seconds: Annotated[float, Field(ge=0, le=MAX_WAIT_SECONDS, description=(
        "Seconds to wait for the job to finish before returning its current state"))] = 30.0,
) -> PipelineJobInfo:
    """Wait for a pipeline or layout job instead of polling get_pipeline_job.
    Returns as soon as the job reaches succeeded, failed or cancelled, or after
    timeout_seconds (at most 50) with the job still queued or running; call it
    again to keep waiting. latest_output shows the newest log line as progress.
    """
    try:
        return _job_info(await _context(ctx).jobs.wait(job_id, timeout=timeout_seconds))
    except PipelineJobError as error:
        raise ToolError(str(error)) from error


async def read_pipeline_log(
    job_id: str,
    ctx: Context[AppContext],
    stream: Literal["stdout", "stderr"] = "stdout",
    offset: Annotated[int, Field(ge=0, description="Byte offset; use the previous page's next_offset to continue")] = 0,
    limit: Annotated[int, Field(ge=1, le=262144, description="Maximum bytes to read from the selected stream")] = 65536,
) -> PipelineLogPage:
    """Read progress or diagnose a pipeline/layout job using its job_id.
    Choose stdout or stderr; offset and limit count bytes. Continue from
    next_offset. eof means the current end of this stream, not job completion;
    use get_pipeline_job for status. Logs belong to this server's retained jobs.
    """
    try:
        payload = await _context(ctx).jobs.read_log(
            job_id,
            stream,
            offset=offset,
            limit=limit,
        )
    except PipelineJobError as error:
        raise ToolError(str(error)) from error
    return PipelineLogPage.model_validate(payload)


async def cancel_pipeline_job(
    job_id: str,
    ctx: Context[AppContext],
) -> PipelineJobInfo:
    """Stop an unwanted pipeline or layout job using its job_id.
    Cancels queued work or terminates a running process tree. Inspect the returned
    status and use get_pipeline_job if still cancelling. Existing output artifacts
    may remain; cancellation does not undo writes.
    """
    try:
        return _job_info(await _context(ctx).jobs.cancel(job_id))
    except PipelineJobError as error:
        raise ToolError(str(error)) from error


async def export_pipeline_settings(tool_id: str, output_path: str | None = None) -> dict[str, Any]:
    """Start from saved pipeline settings when direct parameters are not appropriate.
    tool_id comes from list_pipeline_tools. Creates an editable JSON file with
    inherited saved directories and defaults. Edit this JSON,
    validate it, then pass settings_path to start_pipeline_job. Empty inputs remain
    editable. Explicit output paths must not exist; omitted paths are unique.
    """
    from utilities.Headless_Settings import export_pipeline_settings as export
    try:
        return await asyncio.to_thread(export, tool_id, _PROJECT_ROOT, output_path)
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise ToolError(str(error)) from error


async def export_layout_settings(
    node_fasta_file: Annotated[str | None, Field(description=(
        "Node FASTA for this layout (absolute path recommended); replaces the saved selection"))] = None,
    input_hdf5: Annotated[str | None, Field(description=(
        "Network file for this layout (absolute path recommended); replaces the saved selection"))] = None,
    parameters: Annotated[dict[str, Any] | None, Field(description=(
        "Layout fields to change, e.g. {'TOP_EDGE_PERCENT': 5}; see get_layout_schema"))] = None,
    output_path: str | None = None,
    settings_path: str | None = None,
) -> dict[str, Any]:
    """Export inherited layout settings before start_layout_job.
    The export starts from the user's saved Config preferences (simulation,
    physics, packing, saved inputs), then applies settings_path (an edited JSON
    overlay), then node_fasta_file, input_hdf5 and parameters. Pass the inputs
    when the saved Config has none or holds other files. Execute the returned
    settings_document with start_layout_job. output_path names the new export
    and must not exist; omitted paths are unique. Automatic cache naming is a
    preview, not a reservation. Export does not enqueue work.
    """
    from desktop.Viewer_State import sections
    from utilities.Headless_Settings import export_config_settings as export

    flat = {}
    if parameters:
        flat.update(_layout_overrides(parameters, _EXPORT_RESERVED_KEYS))
    for key, value in (("NODE_FASTA_FILE", node_fasta_file), ("INPUT_HDF5", input_hdf5)):
        if value is not None:
            if not str(value).strip():
                raise ToolError(f"{key.lower()} must be a nonempty path.")
            flat[key] = str(value).strip()
    overlay = None
    if flat:
        overlay = {"schema_version": 2, "kind": "layout"}
        for section, keys in sections("layout").items():
            values = {key: flat[key] for key in keys if key in flat}
            if values:
                overlay[section] = values
    try:
        return await asyncio.to_thread(export, "layout", _PROJECT_ROOT, output_path, settings_path, overlay)
    except (ValueError, TypeError, KeyError, OSError) as error:
        message = str(error)
        if message.startswith(("NODE_FASTA_FILE:", "INPUT_HDF5:")) and "required nonempty path" in message:
            message = (
                f"{message} The saved Config settings select no {message.split(':')[0]}; "
                "pass node_fasta_file and input_hdf5 to export_layout_settings."
            )
        raise ToolError(message) from error


def get_layout_schema() -> dict[str, Any]:
    """Read the layout settings contract before choosing layout parameters.
    Returns every field of the sectioned layout document with its meaning,
    type, built-in default, accepted range and Config GUI range, plus the
    rules start_layout_job enforces and how to choose an edge filter.
    export_layout_settings inherits the user's saved values; individual
    start_layout_job arguments fall back to these built-in defaults.
    """
    from desktop.Viewer_State import sections

    fields = _LAYOUT_CONTRACT["fields"]
    return {
        "schema_version": 2,
        "kind": "layout",
        "sections": {section: list(keys) for section, keys in sections("layout").items()},
        "fields": {
            key: {"section": section, **fields[key]}
            for section, keys in sections("layout").items() for key in keys
        },
        "rules": list(_LAYOUT_CONTRACT["rules"]),
        "choosing_an_edge_filter": list(_LAYOUT_CONTRACT["choosing_an_edge_filter"]),
    }


async def network_statistics(
    input_hdf5: Annotated[str, Field(description=(
        "Network HDF5 file, as for start_layout_job (absolute, or relative to the project root)"))],
    node_fasta_file: Annotated[str | None, Field(description=(
        "Node FASTA the layout will use; only its nodes are counted. Omit to use every network node"))] = None,
    alignment_score: Annotated[Literal["global", "local"], Field(description=(
        "Embedding-alignment networks: score the layout will use (ignored for E-value networks)"))] = "global",
    norm_mode: Annotated[
        Literal["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"],
        Field(description="Embedding-alignment networks: normalization the layout will use"),
    ] = "alignment_length",
    thresholds: Annotated[list[float] | None, Field(max_length=10, description=(
        "Extra SIMILARITY_THRESHOLD values to evaluate"))] = None,
    top_edge_percents: Annotated[list[float] | None, Field(max_length=19, description=(
        "TOP_EDGE_PERCENT values to evaluate instead of the default series (0.1 to 50)"))] = None,
) -> dict[str, Any]:
    """Choose a layout edge filter from evidence instead of guessing.
    Scores the network exactly as start_layout_job would for these nodes and
    settings, and returns the score distribution plus, for a series of
    TOP_EDGE_PERCENT values and any given thresholds, the equivalent
    SIMILARITY_THRESHOLD, kept edges, clusters, isolated nodes and largest
    cluster. Read-only; networks too large to score within about 50 seconds
    return an error suggesting TOP_EDGE_PERCENT.
    """
    from mcp_server.pipeline.Network_Statistics import NetworkStatisticsError, network_statistics as run

    def resolve(value):
        path = os.path.expanduser(os.fspath(value))
        return os.path.abspath(path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path))

    for name, values in (("thresholds", thresholds), ("top_edge_percents", top_edge_percents)):
        for value in values or ():
            if not math.isfinite(value) or (name == "top_edge_percents" and not 0 < value <= 100):
                raise ToolError(f"{name} values must be finite" + (" and in (0, 100]." if name == "top_edge_percents" else "."))
    request = {
        "network_path": resolve(input_hdf5),
        "fasta_path": resolve(node_fasta_file) if node_fasta_file else None,
        "alignment_score": alignment_score,
        "norm_mode": norm_mode,
        "thresholds": list(thresholds or ()),
        "top_edge_percents": list(top_edge_percents) if top_edge_percents else None,
    }
    try:
        return await asyncio.to_thread(run, request)
    except NetworkStatisticsError as error:
        raise ToolError(str(error)) from error


__all__ = [
    "OutputFile",
    "PipelineCatalog",
    "PipelineJobInfo",
    "PipelineJobList",
    "PipelineLogPage",
    "PipelineToolInfo",
    "cancel_pipeline_job",
    "export_layout_settings",
    "export_pipeline_settings",
    "get_compute_capabilities",
    "get_layout_schema",
    "get_pipeline_job",
    "get_pipeline_tool_schema",
    "inspect_pipeline_file",
    "list_pipeline_jobs",
    "list_pipeline_tools",
    "network_statistics",
    "read_pipeline_log",
    "start_layout_job",
    "start_pipeline_job",
    "validate_pipeline_settings",
    "wait_pipeline_job",
]
