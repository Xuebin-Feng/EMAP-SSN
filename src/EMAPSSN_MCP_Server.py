# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Three workflow entry points for the local STDIO MCP server."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.types import ToolAnnotations
from pydantic import Field
from utilities.Application_Identity import PRODUCT_NAME
from mcp_server.Workflow_Operations import AppContext, app_lifespan
from mcp_server.Workflow_Dispatch import dispatch, PipelineAction, ViewerDataAction, ViewerControlAction

MCP_SERVER_VERSION = "0.9.0"

def _load_agent_instructions():
    path = Path(__file__).resolve().parent / "mcp_server" / "Agent_Instructions.md"
    try:
        instructions = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"Cannot load MCP agent guide at {path}. Restore a readable UTF-8 Agent_Instructions.md and restart the server.") from error
    if not instructions.strip():
        raise RuntimeError(f"MCP agent guide at {path} is empty. Restore Agent_Instructions.md and restart the server.")
    return instructions


mcp = MCPServer(
    "emap-ssn", title=PRODUCT_NAME,
    description="Run pipelines and layout jobs, inspect Viewer data, and control Viewer sessions.",
    instructions=_load_agent_instructions(), version=MCP_SERVER_VERSION,
    lifespan=app_lifespan, log_level="WARNING",
)
_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True)


@mcp.tool(title="EMAP-SSN pipelines and layout caches", annotations=_MUTATING, structured_output=True)
async def emapssn_pipeline(ctx: Context[AppContext], action: PipelineAction,
                           arguments: dict[str, Any] = Field(default_factory=dict)) -> dict[str, Any]:
    """Discover, configure, run and monitor scientific pipelines and layout-cache jobs.
    Use action='help' for the catalog or action='describe', arguments={'action': name}
    for one action's schema, effects and example. Jobs can create/overwrite files;
    cancel_job stops queued/running work. Layout generation does not launch a Viewer.
    """
    return await dispatch("emapssn_pipeline", action, arguments, ctx)


@mcp.tool(title="EMAP-SSN Viewer data", annotations=_READ_ONLY, structured_output=True)
async def emapssn_viewer_data(ctx: Context[AppContext], action: ViewerDataAction,
                              arguments: dict[str, Any] = Field(default_factory=dict)) -> dict[str, Any]:
    """Read Viewer sessions, summaries, bounded node pages and captured logs.
    Use help or describe to discover action arguments. Reads share the connection
    established through emapssn_viewer_control; explicit session IDs do not change it.
    """
    return await dispatch("emapssn_viewer_data", action, arguments, ctx)


@mcp.tool(title="EMAP-SSN Viewer control", annotations=_MUTATING, structured_output=True)
async def emapssn_viewer_control(ctx: Context[AppContext], action: ViewerControlAction,
                                 arguments: dict[str, Any] = Field(default_factory=dict)) -> dict[str, Any]:
    """Export/validate settings and launch, connect, disconnect or close Viewers.
    Use help or describe to discover action arguments. Export settings before launch
    to preserve saved preferences. Exports write files; close_session terminates a Viewer.
    Disconnect leaves the Viewer running. Connections are shared with Viewer-data calls.
    """
    return await dispatch("emapssn_viewer_control", action, arguments, ctx)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
