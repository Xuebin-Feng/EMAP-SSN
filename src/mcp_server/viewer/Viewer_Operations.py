# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Viewer data inspection and process lifecycle operation handlers for MCP workflows."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time
from typing import Annotated, Any, Literal

_CORE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _CORE_DIR.parents[1]
_PROJECT_ROOT = str(_CORE_DIR.parents[2])

for _path_str in (str(_SRC_DIR), _PROJECT_ROOT):
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import AfterValidator, BaseModel, Field

from mcp_server.core.App_Context import AppContext, _viewer
from mcp_server.pipeline.Pipeline_Jobs import MAX_WAIT_SECONDS
from mcp_server.viewer.Viewer_Client import DEFAULT_READY_TIMEOUT, MCPViewerError, MCPViewerTimeout
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
    """Capture an immutable snapshot; opt into alignment for residue analysis. Alignment stays in Viewer memory.
    The summary's embedding block reports the optional embedding's model, width and node coverage, never its vectors."""
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
    mode: Literal["normal", "headless", "hidden"] = "normal",
    settings_path: str | None = None,
    settings_document: dict[str, Any] | None = None,
    ready_timeout: Annotated[float, Field(gt=0, le=600)] = DEFAULT_READY_TIMEOUT,
) -> dict[str, Any]:
    """Open a new Viewer after export_config_settings(kind='viewer') and validation.
    Export first to preserve saved visual and other preferences. Consult
    get_viewer_settings_schema, edit only necessary fields in the full export,
    then call validate_viewer_settings. Do not build minimal JSON from defaults.
    Supply exactly one complete settings_document or settings_path, not a cache
    path alone. normal opens a Viewer and terminal; headless opens neither window
    (no OpenGL on Windows and macOS, so no capture_view or print); hidden opens
    neither but renders, so capture_view and print work, and show_session can put
    its window on screen later (with no display it runs headless: the result's
    fallback_reason says why). Waits up to ready_timeout seconds (default 45). status="ready" returns a
    session_id and log paths and connects this transport. Large networks can take
    longer: status="starting" means the Viewer is still loading and was left
    running; follow next_step (wait_viewer_session with its launch_id) rather than
    launching again, and compare phase/cpu_seconds between calls to tell slow from
    stuck. A Viewer that exits first is an error with its log tail. Next use
    get_viewer_summary or read_viewer_log. The Viewer runs independently of backend
    lifetime; use disconnect_viewer_session to leave it running.
    """
    try:
        return await _viewer(ctx).launch_session(
            mode=mode,
            settings_path=settings_path,
            settings_document=settings_document,
            timeout=ready_timeout,
        )
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def show_viewer_window(ctx: Context[AppContext], session_id: str | None = None) -> dict[str, Any]:
    """Put a hidden-mode Viewer's window on screen, on top, without taking the keyboard.
    The taskbar button flashes; the window stays on top (Windows) until the user clicks it.
    window.foreground says whether the platform also gave it keyboard focus, which Windows
    and Wayland may refuse; tell the user to click it. While shown, the Viewer behaves like
    a normal one (dialogs and pages open). Closing the window ends the Viewer. A headless
    Viewer has no window: the error says why (relaunch with mode='hidden')."""
    try:
        result = await _viewer(ctx).window_action("show", session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error
    window = result["window"]
    if window.get("visible") and not window.get("foreground"):
        result["message"] = ("The window is on screen but did not take the keyboard (by design, and the platform "
                             "may refuse); ask the user to click it.")
    return result


async def hide_viewer_window(ctx: Context[AppContext], session_id: str | None = None) -> dict[str, Any]:
    """Take a hidden-mode Viewer's window off screen again; the Viewer keeps running and its
    state, capture_view and print keep working. It then refuses dialogs and pages as before.
    A normal Viewer cannot be hidden, and a headless one has no window."""
    try:
        return await _viewer(ctx).window_action("hide", session_id)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def wait_viewer_session(
    ctx: Context[AppContext],
    launch_id: str,
    ready_timeout: Annotated[float, Field(gt=0, le=600)] = DEFAULT_READY_TIMEOUT,
) -> dict[str, Any]:
    """Keep waiting for a start_viewer_session launch that returned status="starting".
    Supply its launch_id. Waits up to ready_timeout seconds (default 45), then
    returns the same results as start_viewer_session: status="ready" connects this
    transport; status="starting" means still loading, so call again. Never starts or
    stops a Viewer; a launch that exited is an error with its log tail. Works after
    a backend restart. To abandon a launch, use close_viewer_session with launch_id.
    """
    try:
        return await _viewer(ctx).wait_for_launch(launch_id, timeout=ready_timeout)
    except MCPViewerError as error:
        raise ToolError(str(error)) from error


async def close_viewer_session(
    ctx: Context[AppContext],
    session_id: str | None = None,
    launch_id: str | None = None,
) -> dict[str, Any]:
    """Terminate a Viewer when the user intends to close it.
    Omit session_id for this transport's selection or supply a live UUID/alias.
    Alternatively supply the launch_id from start_viewer_session to close that
    launch's Viewer, including one still loading before it publishes a session.
    Returns closed only after verified process exit and descriptor cleanup.
    To leave the Viewer running, use disconnect_viewer_session instead.
    """
    try:
        return await _viewer(ctx).close_session(session_id, launch_id=launch_id)
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
    alignment/embedding/display for Viewer, generation/simulation/physics for layout.
    Change only requested fields and dependencies in the exported JSON file,
    then execute the full export; do not rebuild minimal JSON from schema defaults.
    kind selects the contract; settings_path optionally supplies an edited JSON
    overlay, while output_path selects the new export destination.
    Returns settings and cache paths with inherited Config settings.
    Layout exports select the next automatic version (a preview, not a reservation).
    Viewer exports include only inputs, alignment, embedding, visualization and
    directories, with schema_version=2 and kind=viewer. Layout exports use kind=layout and
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
    A reference absent from the selected MSA is allowed: the Viewer keeps the MSA,
    warns in its startup log, and uses occupancy numbering with offsets inactive.
    The configured reference and offset are preserved; valid does not guarantee
    reference-based numbering. A nonempty reference still requires MSA_FILE.
    embedding.EMBED_FILE is optional: empty or absent selects no embedding. A
    selected file must be a complete embedding database the pipeline wrote
    (only its manifest is checked); covering only some nodes is allowed, and
    nodes it lacks, or holds with another sequence, are left out of
    embedding commands. Next launch with that document. This does not create a Viewer or prove
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


def _cache_choices(cache_path, limit=20):
    """Describe the selected cache and its folder's other caches by saved state."""
    import h5py
    from datetime import datetime, timezone
    from mcp_server.pipeline.Pipeline_File_Inspection import layout_cache_contents

    def describe(path):
        try:
            with h5py.File(path, "r") as hf:
                headers = hf.get("headers")
                contents, problem = layout_cache_contents(hf, len(headers) if headers is not None else 0)
        except (OSError, ValueError, TypeError) as error:
            return {"error": str(error)[:200]}
        return {**contents, "error": problem} if problem else contents

    selected = Path(cache_path)
    others = sorted((path for path in selected.parent.iterdir()
                     if path.is_file() and path.suffix.lower() == ".h5" and path != selected),
                    key=lambda path: (-path.stat().st_mtime_ns, path.name))
    summaries = []
    for path in others[:limit]:
        contents = describe(path)
        summaries.append({"cache_filename": path.name,
                          "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                          **{key: contents.get(key) for key in
                             ("cluster_labels", "group_labels", "last_cluster_params", "error") if key in contents}})
    result = {"cache_contents": describe(selected), "other_caches": summaries}
    if len(others) > limit:
        result["other_caches_omitted"] = len(others) - limit
    return result


async def export_viewer_settings(output_path: str | None = None, settings_path: str | None = None) -> dict[str, Any]:
    """Export inherited Viewer settings before validate_viewer_settings and launch.
    Preserve saved preferences by editing the full export. settings_path optionally
    supplies an edited JSON overlay, including an explicit TARGET_CACHE_PATH;
    otherwise the newest compatible cache is selected. output_path must not exist;
    omitted output paths are unique. Export writes settings but does not launch.
    cache_contents reports the selected cache's saved cluster labels, groups and
    metadata columns; other_caches lists the folder's other caches (newest first)
    by the same state, so a saved figure cache can be selected with an overlay.
    """
    result = await export_config_settings("viewer", output_path, settings_path)
    if result.get("cache_path"):
        try:
            result.update(await asyncio.to_thread(_cache_choices, result["cache_path"]))
        except OSError as error:
            result["cache_contents"] = {"error": str(error)[:200]}
    return result


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
    "hide_viewer_window",
    "list_viewer_sessions",
    "query_viewer_nodes",
    "read_viewer_log",
    "read_viewer_value",
    "show_viewer_window",
    "start_viewer_session",
    "summarize_viewer_subset",
    "validate_viewer_settings",
    "wait_viewer_session",
]


class ViewerBusy(ToolError):
    """The Viewer did not answer in time. What was asked may still happen there; a wait goes on regardless."""


async def _portal_call(ctx, action, arguments, session_id):
    try:
        return await _viewer(ctx).command_action(action, arguments, session_id)
    except MCPViewerTimeout as error:
        raise ViewerBusy(str(error)) from error
    except MCPViewerError as error:
        raise ToolError(str(error)) from error

# Submission ids name a request in the whole Viewer, whoever submitted it: a second client that reuses an id with
# the same payload is silently handed the first client's request. So every different submission needs an id no
# other client will pick, which a random UUID (32 hex digits, 36 with dashes) is. Length is all a server can check;
# the portal's own limit is 128.
SUBMISSION_ID_LENGTH = (32, 128)


def _random_submission_id(value):
    low, high = SUBMISSION_ID_LENGTH
    if not low <= len(value) <= high:
        raise ValueError(
            f"submission_id must be a new random UUID (or other random text) of {low} to {high} characters, "
            f"not {value[:40]!r}: ids are shared by every client of a Viewer. Use a different one for each "
            "different submission, and the same one only to retry an identical payload.")
    return value


SubmissionId = Annotated[str, Field(
    description=("A new random UUID for every different submission, such as one from uuid4(); reuse it only to retry "
                 "the same commands. Ids are shared by every client of the Viewer."),
    json_schema_extra={'minLength': SUBMISSION_ID_LENGTH[0], 'maxLength': SUBMISSION_ID_LENGTH[1]}),
    AfterValidator(_random_submission_id)]


async def execute_viewer_commands(ctx: Context[AppContext], submission_id: SubmissionId,
        commands: str | list[str], session_id: str | None = None) -> dict[str, Any]:
    """Submit existing user commands, ordered and stopped on failure. Give each different submission its own random UUID as submission_id; reuse it only to retry the same commands. Follow next_step: wait_command_request until the request ends, never polling get_command_request in a loop; accepted is not completed. May write files, open dialogs, and start external work."""
    result = await _portal_call(ctx, 'execute_commands', dict(submission_id=submission_id, commands=commands), session_id)
    return result | {'next_step': _wait_step(result)}

async def get_command_request(ctx: Context[AppContext], request_id: str | None = None,
        session_id: str | None = None, offset: Annotated[int, Field(ge=0)] = 0,
        limit: Annotated[int, Field(ge=1, le=100)] = 25, command_id: str | None = None,
        artifact_offset: Annotated[int, Field(ge=0)] = 0, artifact_limit: Annotated[int, Field(ge=1, le=100)] = 25,
        submission_id: str | None = None) -> dict[str, Any]:
    """Read by exactly one request_id or submission_id in the selected Viewer, once. Page outcomes, messages, jobs and artifacts. status indicates execution completion; complete only describes pagination. To follow a request until it ends use wait_command_request, not repeated reads. Awaiting input requires interaction in the visible Viewer."""
    return await _portal_call(ctx, 'get_command_request', dict(request_id=request_id, offset=offset, limit=limit, command_id=command_id, artifact_offset=artifact_offset, artifact_limit=artifact_limit, submission_id=submission_id), session_id)


# The states of a finished command or request (tests pin them to Viewer_Command_Portal.TERMINAL). A wait
# also ends on awaiting_user_input, which only the user at the visible Viewer can end.
FINISHED_STATES = frozenset({'succeeded', 'failed', 'cancelled', 'skipped'})
REQUEST_ENDED = FINISHED_STATES | {'awaiting_user_input'}
# How often the Viewer is asked while waiting: quickly at first, since light commands end within a
# fraction of a second, then less often (each read is two small HTTP requests).
WAIT_POLL_SECONDS = ((2.0, 0.1), (10.0, 0.25), (None, 0.5))
# A progress notification when what the request reports changes, at most this often, and a heartbeat
# when nothing changed for this long, so a client that ends calls with no news keeps waiting.
PROGRESS_MIN_SECONDS = 1.0
HEARTBEAT_SECONDS = 10.0
_now = time.monotonic


async def _pause(seconds):
    await asyncio.sleep(seconds)


def _poll_seconds(elapsed):
    for until, seconds in WAIT_POLL_SECONDS:
        if until is None or elapsed < until:
            return seconds


def _wait_step(record):
    """next_step for a request that has not ended: wait for it, on the Viewer that holds it."""
    return {'tool': 'emapssn_viewer_data', 'action': 'wait_command_request',
            'arguments': {'request_id': record['request_id'], 'session_id': record['session_id']}}


def _wait_line(record):
    """One line saying where a request stands: how many commands have finished, the one at work, its jobs and their progress."""
    commands = record.get('commands', [])
    total = record.get('total_commands', len(commands))
    unfinished = [command for command in commands if command['status'] not in FINISHED_STATES]
    line = f"{record['status']}: {len(commands) - len(unfinished)} of {total} commands finished"
    # The command at work, else the next one waiting.
    current = next((command for command in unfinished if command['status'] != 'queued'), unfinished[0] if unfinished else None)
    if current is not None:
        text = current['command'] if len(current['command']) <= 60 else current['command'][:57] + '...'
        jobs = '; '.join(
            f"job {job['job_id']} {job['status']}" + (f" at {job['position']}" if job.get('position') else '')
            + (f", {job['progress']}" if job.get('progress') else '')
            for job in current.get('jobs', []) if job['status'] not in FINISHED_STATES)
        line += f"; {current['status']}: {text}" + (f" ({jobs})" if jobs else '')
    return line


async def wait_command_request(ctx: Context[AppContext], request_id: str | None = None,
        session_id: str | None = None, submission_id: str | None = None,
        timeout_seconds: Annotated[float, Field(ge=0, le=MAX_WAIT_SECONDS, description=(
            "Seconds to wait for the request to end before returning its current state"))] = 30.0,
        offset: Annotated[int, Field(ge=0)] = 0, limit: Annotated[int, Field(ge=1, le=100)] = 100,
        command_id: str | None = None, artifact_offset: Annotated[int, Field(ge=0)] = 0,
        artifact_limit: Annotated[int, Field(ge=1, le=100)] = 25) -> dict[str, Any]:
    """Wait for a command request instead of polling get_command_request, by exactly one request_id or submission_id.
    Returns the request as get_command_request does as soon as its status is succeeded, failed, cancelled, skipped or
    awaiting_user_input (the user must act in the visible Viewer), or after timeout_seconds (at most 50) with waiting=true
    while it is still queued or running: call again to keep waiting; next_step says how. Sends progress notifications
    to a client that asks for them. Report completion only when status is succeeded."""
    started = _now()
    arguments = dict(request_id=request_id, offset=offset, limit=limit, command_id=command_id,
                     artifact_offset=artifact_offset, artifact_limit=artifact_limit, submission_id=submission_id)
    record, busy, last_line, last_sent = None, False, None, None
    while True:
        try:
            record, busy = await _portal_call(ctx, 'get_command_request', arguments, session_id), False
        except ViewerBusy:
            # A busy Viewer is what a wait is for: ask again until the deadline, with the last record
            # in hand; with none, the error is all there is to report.
            if record is None and _now() - started >= timeout_seconds:
                raise
            busy = True
        except ToolError as error:
            if 'Unknown submission_id' in str(error):
                raise ToolError(f"{error}. It was never queued here, so execute_commands with the same "
                                "submission_id is safe to send again.") from error
            raise
        elapsed = _now() - started
        done = not busy and record['status'] in REQUEST_ENDED
        if done or elapsed >= timeout_seconds:
            record['waited_seconds'] = round(elapsed, 2)
            record['waiting'] = not done
            if busy:
                record['viewer_busy'] = True
            if not done:
                record['next_step'] = _wait_step(record)
            return record
        line = (_wait_line(record) if record is not None else 'waiting') + (' (the Viewer is busy)' if busy else '')
        if last_sent is None or (line != last_line and elapsed - last_sent >= PROGRESS_MIN_SECONDS) \
                or elapsed - last_sent >= HEARTBEAT_SECONDS:
            last_line, last_sent = line, elapsed
            await ctx.report_progress(progress=round(elapsed, 2), total=timeout_seconds, message=line)
        await _pause(min(_poll_seconds(elapsed), max(timeout_seconds - elapsed, 0.0)))

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
    """List existing commands, syntax and effects, or supply command (for example reset) to read detailed source help. Read-only; execute_commands is not required. Needs no Viewer: with none selected it reads this installation's command sources (source="installation"). Each command's requires names the optional inputs (alignment, embedding) it cannot run without; check the snapshot summary's alignment and embedding blocks before running one."""
    if session_id is None and _viewer(ctx).connected_session_id is None:
        # The catalog is parsed from command sources, not live Viewer state; a
        # selected Viewer still answers, since its running code may differ.
        from desktop.Viewer_Inspection import command_catalog
        try:
            catalog = await asyncio.to_thread(command_catalog, command)
        except ValueError as error:
            raise ToolError(str(error)) from error
        return {**catalog, 'source': 'installation'}
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
