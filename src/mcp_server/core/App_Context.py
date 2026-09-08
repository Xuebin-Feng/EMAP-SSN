# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Shared lifecycle context, job managers, and viewer connection state for MCP workflows."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from weakref import WeakKeyDictionary, ref

_CORE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _CORE_DIR.parents[1]
_PROJECT_ROOT = _CORE_DIR.parents[2]

for _path_str in (str(_SRC_DIR), str(_PROJECT_ROOT)):
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from mcp.server import MCPServer
from mcp.server.mcpserver import Context

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_server.pipeline.Pipeline_Jobs import PipelineJobManager
    from mcp_server.viewer.Viewer_Client import ViewerClient


@dataclass
class AppContext:
    jobs: PipelineJobManager
    viewer_connections: WeakKeyDictionary = field(default_factory=WeakKeyDictionary)


@asynccontextmanager
async def app_lifespan(_server: MCPServer) -> AsyncIterator[AppContext]:
    from mcp_server.pipeline.Pipeline_Jobs import PipelineJobManager
    jobs = PipelineJobManager(str(_PROJECT_ROOT), max_pending=16, history_limit=100)
    await jobs.start()
    context = AppContext(jobs=jobs)
    try:
        yield context
    finally:
        await asyncio.gather(*(client.disconnect_session() for client in list(context.viewer_connections.values())))
        context.viewer_connections.clear()
        await jobs.close()


def _context(ctx: Context[AppContext]) -> AppContext:
    return ctx.request_context.lifespan_context


def _viewer(ctx: Context[AppContext]) -> ViewerClient:
    from mcp_server.viewer.Viewer_Client import ViewerClient
    app = _context(ctx)
    transport = ctx.session._connection.outbound
    if transport not in app.viewer_connections:
        transport_ref = ref(transport)
        def transport_closed():
            current = transport_ref()
            return current is None or getattr(current, "_closed", False)
        app.viewer_connections[transport] = ViewerClient(project_root=str(_PROJECT_ROOT), transport_closed=transport_closed)
    return app.viewer_connections[transport]


__all__ = ["AppContext", "_PROJECT_ROOT", "_context", "_viewer", "app_lifespan"]
