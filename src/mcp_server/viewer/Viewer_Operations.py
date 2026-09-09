# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Viewer data inspection and process lifecycle operation handlers for MCP workflows."""

from __future__ import annotations

import asyncio
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

from mcp_server.core.App_Context import AppContext, _viewer
from mcp_server.viewer.Viewer_Client import MCPViewerError
from desktop.Viewer_State import (
    get_viewer_settings_schema as viewer_settings_schema,
    read_viewer_settings,
    validate_viewer_document,
    ViewerSettingsError,
)


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


async def list_viewer_sessions(
    ctx: Context[AppContext],
    offset: Annotated[int, Field(ge=0)] = 0,
    limit: Annotated[int, Field(ge=1, le=100)] = 25,
    max_bytes: Annotated[int, Field(ge=1024, le=65536)] = 16384,
) -> dict[str, Any]:
    """Page live session identities without selecting one or allocating snapshots. Discovery can change between pages."""
    try:
        return await _viewer(ctx).list_sessions(offset=offset, limit=limit, max_bytes=max_bytes)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def get_viewer_summary(
    ctx: Context[AppContext],
    session_id: str | None = None,
    include_visual: bool = False,
    max_bytes: Annotated[int, Field(ge=1024, le=65536)] = 16384,
    include_alignment: bool = False,
) -> dict[str, Any]:
    """Capture an immutable snapshot; opt into alignment for residue analysis. Alignment stays in Viewer memory."""
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
    """Intersect scope with shared Boolean predicates, including residues on alignment snapshots; no files or commands."""
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
    visual_fields: list[Literal["position", "color", "size"]] | None = None,
    session_id: str | None = None,
    cursor: str | None = None,
    limit: Annotated[int, Field(ge=1, le=500)] = 25,
    max_bytes: Annotated[int, Field(ge=1024, le=65536)] = 16384,
    fields: list[Literal["index", "node_id", "visible", "selected", "cluster", "groups", "metadata", "visual"]] | None = None,
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


async def export_viewer_settings(output_path: str | None = None, settings_path: str | None = None) -> dict[str, Any]:
    """Export inherited Viewer settings before validate_viewer_settings and launch.
    Preserve saved preferences by editing the full export. settings_path optionally
    supplies an edited JSON overlay, including an explicit TARGET_CACHE_PATH;
    otherwise the newest compatible cache is selected. output_path must not exist;
    omitted output paths are unique. Export writes settings but does not launch.
    """
    return await export_config_settings("viewer", output_path, settings_path)


__all__ = [
    "ViewerSessionInfo",
    "ViewerSessionList",
    "close_viewer_session",
    "connect_viewer_session",
    "create_viewer_subset",
    "describe_viewer_fields",
    "disconnect_viewer_session",
    "export_config_settings",
    "export_viewer_settings",
    "get_viewer_settings_schema",
    "get_viewer_summary",
    "list_viewer_sessions",
    "query_viewer_nodes",
    "read_viewer_log",
    "read_viewer_value",
    "start_viewer_session",
    "summarize_viewer_subset",
    "validate_viewer_settings",
]


async def _portal_call(ctx, action, arguments, session_id):
    try:
        return await _viewer(ctx).command_action(action, arguments, session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error

async def execute_viewer_commands(ctx: Context[AppContext], submission_id: str,
        commands: str | list[str], session_id: str | None = None) -> dict[str, Any]:
    """Submit existing user commands, ordered and stopped on failure. Reuse submission_id on retries. Poll get_command_request; accepted is not completed. May write files, open dialogs, and start external work."""
    result = await _portal_call(ctx, 'execute_commands', dict(submission_id=submission_id, commands=commands), session_id)
    return result | {'next_step': {'tool': 'emapssn_viewer_data',
        'action': 'get_command_request', 'arguments': {
            'request_id': result['request_id'], 'session_id': result['session_id']}}}

async def get_command_request(ctx: Context[AppContext], request_id: str | None = None,
        session_id: str | None = None, offset: Annotated[int, Field(ge=0)] = 0,
        limit: Annotated[int, Field(ge=1, le=100)] = 25, command_id: str | None = None,
        artifact_offset: Annotated[int, Field(ge=0)] = 0, artifact_limit: Annotated[int, Field(ge=1, le=100)] = 25,
        submission_id: str | None = None) -> dict[str, Any]:
    """Read by exactly one request_id or submission_id in the selected Viewer. Page outcomes, messages, jobs and artifacts. status indicates execution completion; complete only describes pagination. Awaiting input requires interaction in the visible Viewer."""
    return await _portal_call(ctx, 'get_command_request', dict(request_id=request_id, offset=offset, limit=limit, command_id=command_id, artifact_offset=artifact_offset, artifact_limit=artifact_limit, submission_id=submission_id), session_id)

async def list_command_requests(ctx: Context[AppContext], session_id: str | None = None,
        offset: Annotated[int, Field(ge=0)] = 0, limit: Annotated[int, Field(ge=1, le=100)] = 25) -> dict[str, Any]:
    """Recover Viewer-owned request IDs after reconnecting. Completed history is bounded."""
    return await _portal_call(ctx, 'list_command_requests', dict(offset=offset, limit=limit), session_id)

async def read_command_output(ctx: Context[AppContext], request_id: str | None = None,
        session_id: str | None = None, stream: Literal['stdout', 'stderr'] = 'stdout',
        offset: Annotated[int, Field(ge=0)] = 0,
        limit: Annotated[int, Field(ge=4, le=32768)] = 8192,
        submission_id: str | None = None) -> dict[str, Any]:
    """Read by exactly one request_id or submission_id in the selected Viewer. Output uses byte cursors; eof is not execution completion. Inspect truncation and available_from."""
    return await _portal_call(ctx, 'read_command_output', dict(request_id=request_id, stream=stream, offset=offset, limit=limit, submission_id=submission_id), session_id)

async def capture_view(ctx: Context[AppContext], session_id: str | None = None,
        request_id: str | None = None) -> dict[str, Any]:
    """Capture the current canvas with HUD as a PNG preview, at most 1600 pixels per dimension. Optional completed request association is not a historical-state guarantee."""
    return await _portal_call(ctx, 'capture_view', dict(request_id=request_id), session_id)

async def get_command_catalog(ctx: Context[AppContext], command: str | None = None,
        session_id: str | None = None) -> dict[str, Any]:
    """List existing commands, syntax and effects, or supply command (for example reset) to read detailed source help. Read-only; execute_commands is not required."""
    return await _portal_call(ctx, 'get_command_catalog', dict(command=command), session_id)


async def get_residue_distribution(
    ctx: Context[AppContext], snapshot_id: str,
    positions: Annotated[list[str], Field(min_length=1, max_length=100)],
    subset_id: str | None = None, group_by: Literal["none", "cluster", "group"] = "none",
    session_id: str | None = None, cursor: str | None = None,
    limit: Annotated[int, Field(ge=1, le=500)] = 25,
    max_bytes: Annotated[int, Field(ge=1024, le=65536)] = 16384,
) -> dict[str, Any]:
    """Count residues in frozen network populations; requires include_alignment=true. Gaps dilute fractions; unmapped nodes are excluded. Positions are explicit displayed labels, not ranges."""
    arguments = dict(locals())
    arguments.pop("ctx")
    arguments.pop("session_id")
    try:
        return await _viewer(ctx).inspect_data("get_residue_distribution", arguments, session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error
