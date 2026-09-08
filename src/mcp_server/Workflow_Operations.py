# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Reusable operation handlers and connection/job lifecycle for MCP workflows."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary, ref
import os
import json
import asyncio
import sys
from typing import Annotated, Any, Literal

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from mcp_server.MCP_Pipeline_Jobs import (
    PipelineJobError,
    PipelineJobManager,
)
from mcp_server.MCP_Viewer_Client import MCPViewerClient, MCPViewerError
from desktop.Viewer_State import (
    get_viewer_settings_schema as viewer_settings_schema,
    read_viewer_settings, validate_viewer_document, ViewerSettingsError,
)
from tools.tool_helpers.Tool_Pipeline import list_tool_specs
from mcp_server.Pipeline_Settings import (
    DESCRIPTIONS,
    PipelineSettingsError,
    get_pipeline_schema,
    normalize_pipeline_settings,
)


class PipelineToolInfo(BaseModel):
    tool_id: str
    description: str
    script_name: str
    settings_section: str
    required_directories: list[str]
    output_directories: list[str]


class PipelineCatalog(BaseModel):
    tools: list[PipelineToolInfo]
    max_running: int
    max_pending: int


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


class ViewerSessionInfo(BaseModel):
    session_id: str
    session_alias: str | None = None
    cache_metadata: dict[str, Any] | None = None
    metadata_error: str | None = None
    pid: int
    started_at: str


class ViewerSessionList(BaseModel):
    sessions: list[ViewerSessionInfo]
    automatic_selection: bool
    connected_session_id: str | None = None


@dataclass
class AppContext:
    jobs: PipelineJobManager
    viewer_connections: WeakKeyDictionary = field(default_factory=WeakKeyDictionary)


@asynccontextmanager
async def app_lifespan(_server: MCPServer) -> AsyncIterator[AppContext]:
    jobs = PipelineJobManager(_PROJECT_ROOT, max_pending=16, history_limit=100)
    await jobs.start()
    context = AppContext(jobs=jobs)
    try:
        yield context
    finally:
        await asyncio.gather(*(client.disconnect_session() for client in list(context.viewer_connections.values())))
        context.viewer_connections.clear()
        await jobs.close()


def _context(ctx: Context[AppContext]):
    return ctx.request_context.lifespan_context


def _viewer(ctx: Context[AppContext]):
    # MCP 2.1.1 modern STDIO builds per-request Session AND Connection proxies.
    # The standalone outbound channel is shared by requests on one transport.
    app = _context(ctx)
    transport = ctx.session._connection.outbound
    if transport not in app.viewer_connections:
        transport_ref = ref(transport)
        # Both installed MCP dispatchers mark _closed on transport teardown.
        def transport_closed():
            current = transport_ref()
            return current is None or getattr(current, "_closed", False)
        app.viewer_connections[transport] = MCPViewerClient(project_root=_PROJECT_ROOT, transport_closed=transport_closed)
    return app.viewer_connections[transport]


def _job_info(payload):
    return PipelineJobInfo.model_validate(payload)


def list_pipeline_tools() -> PipelineCatalog:
    """Choose a pipeline when its ID is unknown. Returns tool_id values,
    descriptions, directory contracts, and queue capacity. These IDs are arguments,
    not MCP tool names. Next call get_pipeline_tool_schema with the chosen tool_id.
    Layout calculation uses start_layout_job separately.
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
            )
            for spec in list_tool_specs()
        ],
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
    from mcp_server.Compute_Capabilities import discover_compute_capabilities
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
    Relative paths use the project root. Plain BLAST text may require file_type.
    Optional tool_id/parameters supplies inspection context (e.g. BLAST columns).
    Structure, generation completion, and pair coverage are separate conclusions;
    valid structure is not proof of numerical correctness or job readiness.
    Use the findings to prepare settings or qualify the reported result; this
    tool does not search directories or repair files.
    """
    from mcp_server.Pipeline_File_Inspection import inspect_pipeline_file as inspect_file
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
    get_pipeline_job and read_pipeline_log. Files may be created or overwritten
    according to settings; backend exit cancels this server's jobs.
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
        Field(description="Target compute device: 'auto', 'cpu', 'cuda:N', or 'mps'"),
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
    Export first to inherit saved simulation and physics settings, then change
    only requested fields and required dependencies. Individual arguments can
    replace omitted preferences with built-in defaults; use the exported document.
    Validate and enqueue into the shared pipeline FIFO queue; this is separate
    from the pipeline IDs in list_pipeline_tools and has no standalone validator.
    Calculates 2D node coordinates via iterative force-directed physics or UMAP dimension
    reduction and publishes an HDF5 layout cache file along with a canonical FASTA backup and
    manifest. Supply either individual parameters, settings_document, or settings_path.
    Missing defaults and directory paths inherit from EMAP-SSN configuration.
    Follow the returned job_id with get_pipeline_job and read_pipeline_log;
    after success inspect the cache before preparing complete Viewer settings.
    This operation does not launch a Viewer.
    """
    from Layout_Cache_Generator import LayoutGenerationSettings, LayoutGenerationError
    import EMAPSSN_Config as cfg

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
        saved_layout_dir = dirs.get("SAVED_LAYOUT_DIR") or getattr(cfg, "SAVED_LAYOUT_DIR", "Cache_Files/Saved_Layouts")
        if isinstance(saved_layout_dir, str) and not os.path.isabs(saved_layout_dir):
            saved_layout_dir = os.path.join(_PROJECT_ROOT, saved_layout_dir)

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
            "MAX_STEPS": 10000,
            "RMSD_THRESHOLD": 0.005,
            "PERCENTAGE_DROP_THRESHOLD": 0.1,
            "RMSD_WINDOW": 50,
            "ENABLE_PROGRESSIVE_SIMULATION": False,
            "PACKING_GEOMETRY": "Square",
            "PACKING_GRID_SIZE": 20.0,
            "BOX_SCALE": 2.0,
            "PACKING_PADDING": 10.0,
            "MAX_FORCE_LIMIT": 20.0,
            "MAX_TOTAL_REPULSION_FORCE": 0.0,
        }
        if parameters and isinstance(parameters, dict):
            for k, v in parameters.items():
                payload[k.upper()] = v

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
    Returns status, failure_message, and output locations. Queued/running is not
    success; wait for a terminal status. On failure use read_pipeline_log for both
    streams; after succeeded use inspect_pipeline_file on relevant output files.
    """
    try:
        return _job_info(await _context(ctx).jobs.get_job(job_id))
    except PipelineJobError as error:
        raise ToolError(str(error)) from error


async def read_pipeline_log(
    job_id: str,
    stream: Literal["stdout", "stderr"],
    ctx: Context[AppContext],
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


async def list_viewer_sessions(ctx: Context[AppContext],
                               offset: Annotated[int, Field(ge=0)] = 0,
                               limit: Annotated[int, Field(ge=1, le=100)] = 25,
                               max_bytes: Annotated[int, Field(ge=1024, le=65536)] = 16384) -> dict[str, Any]:
    """Page live session identities without selecting one or allocating snapshots. Discovery can change between pages."""
    try:
        return await _viewer(ctx).list_sessions(offset=offset, limit=limit, max_bytes=max_bytes)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def get_viewer_summary(
    ctx: Context[AppContext],
    session_id: str | None = None,
    max_bytes: Annotated[int, Field(ge=1024, le=65536)] = 16384,
) -> dict[str, Any]:
    """Capture a fresh immutable snapshot; reuse snapshot_id for subsequent reads."""
    arguments = dict(locals())
    arguments.pop("ctx")
    arguments.pop("session_id")
    try:
        return await _viewer(ctx).inspect_data("get_summary", arguments, session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def describe_viewer_fields(
    ctx: Context[AppContext],
    snapshot_id: str,
    session_id: str | None = None,
    cursor: str | None = None,
    limit: Annotated[int, Field(ge=1, le=500)] = 25,
    max_bytes: Annotated[int, Field(ge=1024, le=65536)] = 16384,
) -> dict[str, Any]:
    """Page snapshot metadata types, missingness and provenance availability."""
    arguments = dict(locals())
    arguments.pop("ctx")
    arguments.pop("session_id")
    try:
        return await _viewer(ctx).inspect_data("describe_fields", arguments, session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def create_viewer_subset(
    ctx: Context[AppContext],
    snapshot_id: str,
    scope: Literal["all", "visible", "selected"],
    expression: str | None = None,
    session_id: str | None = None,
    max_bytes: Annotated[int, Field(ge=1024, le=65536)] = 16384,
) -> dict[str, Any]:
    """Intersect explicit scope with metadata/header/label/selection predicates; no commands or file/residue predicates."""
    arguments = dict(locals())
    arguments.pop("ctx")
    arguments.pop("session_id")
    try:
        return await _viewer(ctx).inspect_data("create_subset", arguments, session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def summarize_viewer_subset(
    ctx: Context[AppContext],
    snapshot_id: str,
    subset_id: str | None = None,
    columns: list[str] | None = None,
    session_id: str | None = None,
    cursor: str | None = None,
    limit: Annotated[int, Field(ge=1, le=500)] = 25,
    max_bytes: Annotated[int, Field(ge=1024, le=65536)] = 16384,
) -> dict[str, Any]:
    """Exact metadata and membership statistics; omitted subset uses all nodes. Page complete category counts."""
    arguments = dict(locals())
    arguments.pop("ctx")
    arguments.pop("session_id")
    try:
        return await _viewer(ctx).inspect_data("summarize_subset", arguments, session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def query_viewer_nodes(
    ctx: Context[AppContext],
    snapshot_id: str,
    subset_id: str | None = None,
    columns: list[str] | None = None,
    session_id: str | None = None,
    cursor: str | None = None,
    limit: Annotated[int, Field(ge=1, le=500)] = 25,
    max_bytes: Annotated[int, Field(ge=1024, le=65536)] = 16384,
) -> dict[str, Any]:
    """Page snapshot nodes in original index order. Omitted columns returns no metadata."""
    arguments = dict(locals())
    arguments.pop("ctx")
    arguments.pop("session_id")
    try:
        return await _viewer(ctx).inspect_data("query_nodes", arguments, session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def read_viewer_value(
    ctx: Context[AppContext],
    snapshot_id: str,
    index: Annotated[int, Field(ge=0)],
    field: Literal["node_id", "groups", "metadata"],
    column: str | None = None,
    member_index: Annotated[int | None, Field(ge=0)] = None,
    session_id: str | None = None,
    offset: Annotated[int, Field(ge=0)] = 0,
    limit: Annotated[int, Field(ge=1, le=16384)] = 2048,
    max_bytes: Annotated[int, Field(ge=1024, le=65536)] = 16384,
) -> dict[str, Any]:
    """Read exact JSON-text character slices; concatenate text then JSON-decode. Continue from next_offset."""
    arguments = dict(locals())
    arguments.pop("ctx")
    arguments.pop("session_id")
    try:
        return await _viewer(ctx).inspect_data("read_value", arguments, session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def start_viewer_session(
    ctx: Context[AppContext],
    mode: Literal["normal", "headless"] = "normal",
    settings_path: str | None = None,
    settings_document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Open a new Viewer after export_config_settings(kind='viewer') and validation.
    Export first to preserve saved visual and other preferences. Consult
    get_viewer_settings_schema, edit only necessary fields in the full export,
    then call validate_viewer_settings. Do not build minimal JSON from defaults.
    Supply exactly one complete settings_document or settings_path, not a cache
    path alone. normal opens a Viewer and terminal; headless opens neither window.
    Returns a ready session_id and log paths and connects this transport to it.
    Next use get_viewer_summary or read_viewer_log. The Viewer runs independently
    of backend lifetime; use disconnect_viewer_session to leave it running.
    """
    try:
        return await _viewer(ctx).launch_session(
            mode=mode,
            settings_path=settings_path,
            settings_document=settings_document,
        )
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def close_viewer_session(
    ctx: Context[AppContext],
    session_id: str | None = None,
) -> dict[str, Any]:
    """Terminate a Viewer when the user intends to close it.
    Omit session_id for this transport's selection or supply a live UUID/alias.
    Returns closed only after verified process exit and descriptor cleanup.
    To leave the Viewer running, use disconnect_viewer_session instead.
    """
    try:
        return await _viewer(ctx).close_session(session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


def get_viewer_settings_schema() -> dict[str, Any]:
    """Understand fields when editing a full exported Viewer document.
    First use export_config_settings(kind='viewer') to inherit saved preferences;
    schema defaults are reference values, not replacements for user settings. Next call
    validate_viewer_settings. This contract differs from pipeline/layout documents.
    """
    return viewer_settings_schema()


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


async def export_config_settings(
    kind: Literal["layout", "viewer"], output_path: str | None = None,
    settings_path: str | None = None,
) -> dict[str, Any]:
    """Required first step for a new Viewer or layout: export inherited settings.
    Reads viewer_settings.json and preserves the relevant saved preferences:
    alignment/display for Viewer, generation/simulation/physics for layout.
    Change only requested fields and dependencies in the exported JSON file,
    then execute the full export; do not rebuild minimal JSON from schema defaults.
    kind selects the contract; settings_path optionally supplies an edited JSON
    overlay, while output_path selects the new export destination.
    Returns settings and cache paths with inherited Config settings.
    Layout exports select the next automatic version (a preview, not a reservation).
    Viewer exports include only inputs, alignment, visualization and directories,
    with schema_version=2 and kind=viewer. Layout exports use kind=layout and
    inputs/network/layout/simulation/physics/packing/output sections. Viewer
    generation settings come from verified cache provenance. Exports select the newest compatible cache
    unless an explicit path is supplied in the optional edited JSON overlay.
    Edit the exported file, then execute using settings_path; execution does not
    reload personal settings. Explicit output paths must not already exist.
    Next use validate_viewer_settings then start_viewer_session for a Viewer,
    or start_layout_job for a layout. Exporting does not execute either operation.
    """
    from utilities.Headless_Settings import export_config_settings as export
    try:
        return await asyncio.to_thread(export, kind, _PROJECT_ROOT, output_path, settings_path)
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise ToolError(str(error)) from error


async def validate_viewer_settings(
    settings_document: dict[str, Any] | None = None,
    settings_path: str | None = None,
) -> dict[str, Any]:
    """Check the full export from export_config_settings(kind='viewer') before launch.
    Preserve inherited settings when editing it. Required version 2 sections and
    fields must be explicit. Validation resolves generation settings from verified
    cache metadata internally; those settings are never added to the returned JSON.
    Supply exactly one settings_document or settings_path following
    get_viewer_settings_schema. Checks source files and cache identity and returns
    valid plus a normalized settings_document; invalid settings raise a tool error.
    Next launch with that document. This does not create a Viewer or prove
    numerical/scientific correctness of its inputs.
    """
    try:
        document = read_viewer_settings(settings_document=settings_document, settings_path=settings_path,
                                        project_root=_PROJECT_ROOT)
        normalized = await asyncio.to_thread(validate_viewer_document, document, _PROJECT_ROOT)
        return {"valid": True, "settings_document": normalized}
    except ViewerSettingsError as error:
        raise ToolError(str(error)) from error


async def connect_viewer_session(ctx: Context[AppContext], session_id: str | None = None) -> dict[str, Any]:
    """Select an existing Viewer after list_viewer_sessions identifies the target.
    Use a full UUID or exact eight-character title alias (case-insensitive).
    Ambiguous aliases fail; omit only when exactly one Viewer is running.
    Returns the connected full session_id. Subsequent Viewer calls may omit it;
    next use get_viewer_summary. This operation does not launch a new Viewer.
    """
    try:
        return await _viewer(ctx).connect_session(session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def disconnect_viewer_session(ctx: Context[AppContext]) -> dict[str, Any]:
    """Leave the connected Viewer running while ending this transport's selection.
    Repeated calls are safe; returns the previous session_id or null. Reconnect
    using connect_viewer_session when needed. To terminate the Viewer, use
    close_viewer_session. Retain its full ID for captured log reads after disconnect.
    """
    return await _viewer(ctx).disconnect_session()


async def read_viewer_log(ctx: Context[AppContext], session_id: str | None = None,
                          stream: Literal["stdout", "stderr"] = "stdout",
                          offset: int = 0, limit: int = 8192) -> dict[str, Any]:
    """Read progress or diagnose an MCP-launched Viewer in normal or headless mode.
    Omit session_id for the connected Viewer. stream selects stdout or stderr;
    offset is a nonnegative byte position and limit is 1..1048576 bytes.
    Continue from next_offset; eof is the current stream end, not process exit.
    Supply the full session ID to read retained output after disconnect or close.
    Logs remain on disk; retained IDs are available for this transport's lifetime.
    Viewers launched without MCP output capture do not provide these logs.
    """
    try:
        return await _viewer(ctx).read_log(session_id, stream=stream, offset=offset, limit=limit)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def export_layout_settings(output_path: str | None = None, settings_path: str | None = None) -> dict[str, Any]:
    """Export inherited layout settings before start_layout_job.
    Preserve saved simulation and physics preferences by editing the full export.
    settings_path optionally supplies an edited JSON overlay; output_path names
    the new export and must not exist. Omitted output paths are unique. Automatic
    cache naming is a preview, not a reservation. Export does not enqueue work.
    """
    return await export_config_settings("layout", output_path, settings_path)


async def export_viewer_settings(output_path: str | None = None, settings_path: str | None = None) -> dict[str, Any]:
    """Export inherited Viewer settings before validate_viewer_settings and launch.
    Preserve saved preferences by editing the full export. settings_path optionally
    supplies an edited JSON overlay, including an explicit TARGET_CACHE_PATH;
    otherwise the newest compatible cache is selected. output_path must not exist;
    omitted output paths are unique. Export writes settings but does not launch.
    """
    return await export_config_settings("viewer", output_path, settings_path)
