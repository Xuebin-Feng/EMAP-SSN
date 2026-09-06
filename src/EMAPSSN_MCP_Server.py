# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Local STDIO MCP adapter for SSN pipeline and Viewer inspection services."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
import asyncio
import sys
from typing import Annotated, Any, Literal

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from mcp_server.MCP_Pipeline_Jobs import (
    PipelineJobError,
    PipelineJobManager,
)
from mcp_server.MCP_Viewer_Client import MCPViewerClient, MCPViewerError
from utilities.Application_Identity import PRODUCT_NAME
from utilities.Tool_Execution import list_tool_specs
from mcp_server.Pipeline_Settings import (
    DESCRIPTIONS,
    PipelineSettingsError,
    get_pipeline_schema,
    normalize_pipeline_settings,
)


MCP_SERVER_VERSION = "0.4.0"


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
    pid: int
    started_at: str


class ViewerSessionList(BaseModel):
    sessions: list[ViewerSessionInfo]
    automatic_selection: bool


@dataclass
class AppContext:
    jobs: PipelineJobManager
    viewer: MCPViewerClient


@asynccontextmanager
async def app_lifespan(_server: MCPServer) -> AsyncIterator[AppContext]:
    jobs = PipelineJobManager(_PROJECT_ROOT, max_pending=16, history_limit=100)
    await jobs.start()
    try:
        yield AppContext(jobs=jobs, viewer=MCPViewerClient())
    finally:
        await jobs.close()


mcp = MCPServer(
    "emap-ssn",
    title=PRODUCT_NAME,
    description="Run allowlisted SSN pipelines and inspect local Viewers read-only.",
    instructions=(
        "Discover pipeline parameters with get_pipeline_tool_schema and preview "
        "requests with validate_pipeline_settings. Supply native JSON numbers "
        "and booleans; unknown setting keys are rejected. "
        "Pipeline jobs may create or overwrite files according to the supplied "
        "settings. Jobs and their FIFO queue belong to this STDIO server and are "
        "cancelled when it exits. Viewer tools are read-only."
    ),
    version=MCP_SERVER_VERSION,
    lifespan=app_lifespan,
    log_level="WARNING",
)


_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
_START_JOB = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=True,
)
_CANCEL_JOB = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=True,
    openWorldHint=False,
)


def _context(ctx: Context[AppContext]):
    return ctx.request_context.lifespan_context


def _job_info(payload):
    return PipelineJobInfo.model_validate(payload)


@mcp.tool(
    title="List SSN pipeline tools",
    annotations=_READ_ONLY,
    structured_output=True,
)
def list_pipeline_tools() -> PipelineCatalog:
    """List the fixed pipeline catalog and its directory contracts."""
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


@mcp.tool(title="Get usable compute capabilities", annotations=_READ_ONLY, structured_output=True)
async def get_compute_capabilities(tool_id: str | None = None) -> dict[str, Any]:
    """Discover runtime devices and memory without benchmarks or tensor operations.
    Optionally describe a pipeline's applicable settings. Metadata support is
    unverified by computation; physical devices unavailable to this runtime are omitted.
    """
    from mcp_server.Compute_Capabilities import discover_compute_capabilities
    try:
        return await asyncio.to_thread(discover_compute_capabilities, _PROJECT_ROOT, tool_id)
    except KeyError as error:
        raise ToolError(str(error)) from error


@mcp.tool(title="Inspect a pipeline file", annotations=_READ_ONLY, structured_output=True)
async def inspect_pipeline_file(
    path: str,
    file_type: Literal["auto", "fasta", "alignment_fasta", "embedding", "network", "sparse_msa", "blast_tabular", "settings"] = "auto",
    tool_id: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect a selected file read-only without scanning numerical HDF5 payloads.
    Relative paths use the project root. Plain BLAST text may require file_type.
    Optional tool_id/parameters supplies inspection context (e.g. BLAST columns).
    Structure, generation completion, and pair coverage are separate conclusions;
    valid structure is not proof of numerical correctness or job readiness.
    """
    from mcp_server.Pipeline_File_Inspection import inspect_pipeline_file as inspect_file
    try:
        return await asyncio.to_thread(inspect_file, path, _PROJECT_ROOT, file_type, tool_id, parameters)
    except (KeyError, TypeError, ValueError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(title="Get pipeline settings schema", annotations=_READ_ONLY, structured_output=True)
def get_pipeline_tool_schema(tool_id: str) -> dict[str, Any]:
    """Discover accepted parameters, defaults, choices, conditions, and an example."""
    try:
        return get_pipeline_schema(tool_id, _PROJECT_ROOT)
    except (KeyError, OSError, ValueError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(title="Validate pipeline settings", annotations=_READ_ONLY, structured_output=True)
def validate_pipeline_settings(
    tool_id: str,
    parameters: dict[str, Any] | None = None,
    directories: dict[str, Any] | None = None,
    settings_document: dict[str, Any] | None = None,
    settings_path: str | None = None,
) -> dict[str, Any]:
    """Preview effective settings without creating files or jobs. Supply exactly one
    of parameters, settings_document, or settings_path; directories accompanies
    parameters only. Numbers and booleans require native JSON types. This checks
    configuration, not file existence, model credentials, or hardware readiness.
    """
    return normalize_pipeline_settings(
        tool_id, _PROJECT_ROOT, parameters=parameters, directories=directories,
        settings_document=settings_document, settings_path=settings_path,
    )


@mcp.tool(
    title="Start an SSN pipeline job",
    annotations=_START_JOB,
    structured_output=True,
)
async def start_pipeline_job(
    tool_id: Annotated[str, Field(description="Stable ID from list_pipeline_tools")],
    ctx: Context[AppContext],
    settings_document: Annotated[
        dict[str, Any] | None,
        Field(description="Existing exported SSN settings document"),
    ] = None,
    settings_path: Annotated[
        str | None,
        Field(description="Path to an existing exported SSN settings document"),
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
    """Validate and enqueue a pipeline. Supply exactly one of parameters,
    settings_document, or settings_path. Optional directories goes with parameters.
    Use validate_pipeline_settings for a preview; invalid requests never queue.
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


@mcp.tool(
    title="List SSN pipeline jobs",
    annotations=_READ_ONLY,
    structured_output=True,
)
async def list_pipeline_jobs(
    ctx: Context[AppContext],
    limit: Annotated[int, Field(ge=1, le=100)] = 100,
) -> PipelineJobList:
    """List the newest jobs owned by this STDIO server."""
    try:
        jobs = await _context(ctx).jobs.list_jobs(limit=limit)
    except PipelineJobError as error:
        raise ToolError(str(error)) from error
    return PipelineJobList(jobs=[_job_info(job) for job in jobs])


@mcp.tool(
    title="Get an SSN pipeline job",
    annotations=_READ_ONLY,
    structured_output=True,
)
async def get_pipeline_job(
    job_id: str,
    ctx: Context[AppContext],
) -> PipelineJobInfo:
    """Get current status and output locations for one job."""
    try:
        return _job_info(await _context(ctx).jobs.get_job(job_id))
    except PipelineJobError as error:
        raise ToolError(str(error)) from error


@mcp.tool(
    title="Read an SSN pipeline log",
    annotations=_READ_ONLY,
    structured_output=True,
)
async def read_pipeline_log(
    job_id: str,
    stream: Literal["stdout", "stderr"],
    ctx: Context[AppContext],
    offset: Annotated[int, Field(ge=0)] = 0,
    limit: Annotated[int, Field(ge=1, le=262144)] = 65536,
) -> PipelineLogPage:
    """Read a bounded byte page from a job's captured output stream."""
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


@mcp.tool(
    title="Cancel an SSN pipeline job",
    annotations=_CANCEL_JOB,
    structured_output=True,
)
async def cancel_pipeline_job(
    job_id: str,
    ctx: Context[AppContext],
) -> PipelineJobInfo:
    """Cancel a queued job or terminate a running pipeline process tree."""
    try:
        return _job_info(await _context(ctx).jobs.cancel(job_id))
    except PipelineJobError as error:
        raise ToolError(str(error)) from error


@mcp.tool(
    title="List local EMAP-SSN Viewer sessions",
    annotations=_READ_ONLY,
    structured_output=True,
)
async def list_viewer_sessions(
    ctx: Context[AppContext],
) -> ViewerSessionList:
    """List authenticated live Viewers without exposing discovery secrets."""
    payload = await _context(ctx).viewer.list_sessions()
    return ViewerSessionList.model_validate(payload)


@mcp.tool(
    title="Get EMAP-SSN Viewer summary",
    annotations=_READ_ONLY,
    structured_output=True,
)
async def get_viewer_summary(
    ctx: Context[AppContext],
    session_id: str | None = None,
) -> dict[str, Any]:
    """Read a bounded summary from one live Viewer."""
    try:
        return await _context(ctx).viewer.get_summary(session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


@mcp.tool(
    title="Query EMAP-SSN Viewer nodes",
    annotations=_READ_ONLY,
    structured_output=True,
)
async def query_viewer_nodes(
    ctx: Context[AppContext],
    session_id: str | None = None,
    scope: Literal["all", "visible", "selected"] = "all",
    offset: Annotated[int, Field(ge=0)] = 0,
    limit: Annotated[int, Field(ge=1, le=500)] = 100,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Read an index-ordered page of all, visible, or selected Viewer nodes."""
    try:
        return await _context(ctx).viewer.query_nodes(
            session_id,
            scope=scope,
            offset=offset,
            limit=limit,
            columns=columns,
        )
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
