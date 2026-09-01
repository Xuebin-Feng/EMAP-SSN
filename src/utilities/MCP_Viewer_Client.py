# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Asynchronous client for authenticated read-only Viewer inspection."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
import urllib.error
import urllib.parse
import urllib.request

from utilities.Viewer_Sessions import (
    discover_viewer_sessions,
    select_viewer_session,
)


class MCPViewerError(RuntimeError):
    """Raised when an MCP Viewer inspection request cannot be completed."""


class MCPViewerClient:
    def __init__(self, *, discovery_timeout=0.5, request_timeout=5.0):
        self.discovery_timeout = float(discovery_timeout)
        self.request_timeout = float(request_timeout)

    async def list_sessions(self):
        sessions = await asyncio.to_thread(
            discover_viewer_sessions,
            timeout=self.discovery_timeout,
        )
        return {
            "sessions": [
                {
                    "session_id": session.session_id,
                    "pid": session.pid,
                    "started_at": session.started_at,
                }
                for session in sessions
            ],
            "automatic_selection": len(sessions) == 1,
        }

    async def get_summary(self, session_id=None):
        return await self._get(session_id, "/api/mcp/v1/summary")

    async def query_nodes(
        self,
        session_id=None,
        *,
        scope="all",
        offset=0,
        limit=100,
        columns=None,
    ):
        parameters = {
            "scope": scope,
            "offset": offset,
            "limit": limit,
        }
        if columns is not None:
            parameters["columns"] = ",".join(columns)
        query = urllib.parse.urlencode(parameters)
        return await self._get(
            session_id,
            f"/api/mcp/v1/nodes?{query}",
        )

    async def _get(self, session_id, endpoint):
        try:
            session = await asyncio.to_thread(
                select_viewer_session,
                session_id,
                timeout=self.discovery_timeout,
            )
        except LookupError as error:
            raise MCPViewerError(str(error)) from error
        return await asyncio.to_thread(self._request, session, endpoint)

    def _request(self, session, endpoint):
        request = urllib.request.Request(
            f"{session.base_url}{endpoint}",
            headers={"Authorization": f"Bearer {session.token}"},
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            message = f"Viewer inspection returned HTTP {error.code}."
            try:
                error_payload = json.loads(error.read().decode("utf-8"))
                if isinstance(error_payload, Mapping) and error_payload.get("error"):
                    message = str(error_payload["error"])
            except (OSError, UnicodeError, ValueError):
                pass
            raise MCPViewerError(message) from error
        except (
            OSError,
            UnicodeError,
            ValueError,
            urllib.error.URLError,
        ) as error:
            raise MCPViewerError(f"Could not inspect the Viewer: {error}") from error
        if not isinstance(payload, Mapping):
            raise MCPViewerError("The Viewer returned an invalid JSON response.")
        return dict(payload)


__all__ = ["MCPViewerClient", "MCPViewerError"]
