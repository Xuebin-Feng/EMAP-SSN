# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Authenticated Viewer connection state and independent process lifecycle."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import psutil
from mcp_server.Viewer_Sessions import (
    SESSION_DIRECTORY_ENV, LAUNCH_ID_ENV, session_directory,
    discover_viewer_sessions, remove_viewer_session, select_viewer_session,
)
from utilities.Viewer_Settings import read_viewer_settings, validate_viewer_document, ViewerSettingsError


class MCPViewerError(RuntimeError):
    """Viewer configuration, connection or lifecycle failure."""


def _identity(pid):
    try:
        return psutil.Process(pid)
    except psutil.NoSuchProcess:
        return None


def _alive(process):
    try:
        return process is not None and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _terminate_tree(process, timeout=5.0):
    """Capture descendants before terminating parents; psutil checks PID reuse."""
    if not _alive(process):
        return
    targets = process.children(recursive=True) + [process]
    for target in reversed(targets):
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=timeout)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=timeout)
    if alive:
        raise MCPViewerError("Could not terminate Viewer processes: " + ", ".join(str(p.pid) for p in alive))


class MCPViewerClient:
    def __init__(self, project_root=None, *, discovery_timeout=0.5, request_timeout=5.0):
        self.project_root = os.path.abspath(project_root or Path(__file__).resolve().parents[2])
        self.discovery_timeout = float(discovery_timeout)
        self.request_timeout = float(request_timeout)
        self.connected_session_id = None
        self._selection_lock = asyncio.Lock()

    def _target(self, session_id):
        target = session_id if session_id is not None else self.connected_session_id
        if target is None:
            raise MCPViewerError("No Viewer is connected. Call connect_viewer_session or supply session_id.")
        return target

    async def connect_session(self, session_id=None):
        try:
            session = await asyncio.to_thread(select_viewer_session, session_id, timeout=self.discovery_timeout)
            await asyncio.to_thread(self._request, session, "/api/mcp/v1/summary")
        except LookupError as error:
            raise MCPViewerError(str(error)) from error
        async with self._selection_lock:
            self.connected_session_id = session.session_id
        return {"connected": True, "session_id": session.session_id, "pid": session.pid}

    async def disconnect_session(self):
        async with self._selection_lock:
            previous = self.connected_session_id
            self.connected_session_id = None
        return {"disconnected": True, "session_id": previous}

    async def launch_session(self, *, settings_document=None, settings_path=None, mode="normal", timeout=30.0):
        if mode not in {"normal", "headless"}:
            raise MCPViewerError("mode: expected normal or headless.")
        try:
            document = read_viewer_settings(settings_document=settings_document, settings_path=settings_path,
                                            project_root=self.project_root)
            settings = await asyncio.to_thread(validate_viewer_document, document, self.project_root)
        except ViewerSettingsError as error:
            raise MCPViewerError(str(error)) from error
        launch_id = uuid.uuid4().hex
        directory = Path(session_directory()) / "launches" / launch_id
        directory.mkdir(parents=True, mode=0o700)
        snapshot = directory / "settings.json"
        snapshot.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
        script = Path(self.project_root) / "src" / "EMAPSSN_Viewer.py"
        command = [sys.executable, "-u", str(script), "--settings", str(snapshot), "--delete-settings"]
        env = os.environ.copy()
        for key in ("SSN_VIEWER_SETTINGS_PATH", "SSN_TARGET_CACHE_PATH", "SSN_TARGET_CACHE_MODE", "SSN_TARGET_CACHE",
                    "SSN_VIEWER_HEADLESS", "SSN_VIEWER_EXPLICIT_SETTINGS"):
            env.pop(key, None)
        env[SESSION_DIRECTORY_ENV] = session_directory()
        env[LAUNCH_ID_ENV] = launch_id
        if mode == "headless":
            command.append("--headless")
            env["QT_QPA_PLATFORM"] = "offscreen"
            env["SSN_VIEWER_HEADLESS"] = "1"
        elif env.get("QT_QPA_PLATFORM") == "offscreen":
            env.pop("QT_QPA_PLATFORM")
        options = ({"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB}
                   if sys.platform == "win32" else {"start_new_session": True})
        proc = None
        root_process = None
        session = None
        ready = False
        try:
            # Windows MCP transports terminate their subprocess trees on disconnect.
            # A short-lived broker exits before readiness, severing that ownership
            # chain while recording the independent launcher's exact identity.
            identity_path = directory / "process.json"
            if sys.platform == "win32":
                broker = (
                    "import subprocess,sys,json,psutil; "
                    "p=subprocess.Popen(json.loads(sys.argv[1]),stdin=subprocess.DEVNULL,"
                    "creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP); "
                    "open(sys.argv[2],'w').write(json.dumps({'pid':p.pid,'created':psutil.Process(p.pid).create_time()}))"
                )
                command = [sys.executable, "-c", broker, json.dumps(command), str(identity_path)]
            with stdout_path.open("ab", buffering=0) as out, stderr_path.open("ab", buffering=0) as err:
                proc = subprocess.Popen(command, cwd=self.project_root, env=env, stdin=subprocess.DEVNULL,
                                        stdout=out, stderr=err, **options)
            root_process = _identity(proc.pid)
            if sys.platform == "win32":
                code = await asyncio.to_thread(proc.wait, timeout=10)
                if code != 0:
                    raise MCPViewerError(f"Viewer launch broker failed (code {code}).")
                identity = json.loads(identity_path.read_text())
                root_process = _identity(identity["pid"])
                if root_process is None or root_process.create_time() != identity["created"]:
                    raise MCPViewerError("Independent Viewer launcher exited before readiness.")
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                sessions = await asyncio.to_thread(discover_viewer_sessions, timeout=self.discovery_timeout)
                session = next((item for item in sessions if item.launch_id == launch_id), None)
                if session is not None:
                    await asyncio.to_thread(self._request, session, "/api/mcp/v1/summary")
                    async with self._selection_lock:
                        self.connected_session_id = session.session_id
                    ready = True
                    # A daemon reaps the launcher without owning the independent Viewer lifetime.
                    threading.Thread(target=proc.wait, daemon=True, name="viewer-launcher-reaper").start()
                    return {"status": "ready", "session_id": session.session_id, "pid": session.pid,
                            "base_url": session.base_url, "started_at": session.started_at, "mode": mode,
                            "cache_path": settings["TARGET_CACHE_PATH"],
                            "stdout_log": str(stdout_path), "stderr_log": str(stderr_path)}
                if not await asyncio.to_thread(_alive, root_process):
                    raise MCPViewerError(f"Viewer exited before readiness (code {proc.returncode}).")
                await asyncio.sleep(0.1)
            raise MCPViewerError(f"Timed out after {timeout} seconds waiting for Viewer readiness.")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if isinstance(error, PermissionError) and sys.platform == "win32":
                error = MCPViewerError(
                    "The Windows host denied an independent Viewer process (its Job Object may forbid breakaway). "
                    "Open the Viewer through the GUI or CLI, then use connect_viewer_session."
                )
            tails = []
            for path in (stdout_path, stderr_path):
                if path.exists():
                    with path.open("rb") as handle:
                        handle.seek(max(0, path.stat().st_size - 4096))
                        tails.append(handle.read().decode("utf-8", errors="replace"))
            raise MCPViewerError(f"{error} Logs: {directory}\n" + "\n".join(tails)) from error
        finally:
            if not ready:
                try:
                    if sys.platform == "win32" and identity_path.exists():
                        recorded = json.loads(identity_path.read_text())
                        independent = _identity(recorded["pid"])
                        if independent is not None and independent.create_time() == recorded["created"]:
                            await asyncio.shield(asyncio.to_thread(_terminate_tree, independent))
                    if root_process is not None:
                        await asyncio.shield(asyncio.to_thread(_terminate_tree, root_process))
                    if session is not None:
                        actual = _identity(session.pid)
                        if actual is not None and actual.create_time() == session.process_created_at:
                            await asyncio.shield(asyncio.to_thread(_terminate_tree, actual))
                        remove_viewer_session(session)
                    if proc is not None:
                        await asyncio.to_thread(proc.wait, timeout=5)
                    snapshot.unlink(missing_ok=True)
                except Exception as error:
                    raise MCPViewerError(f"Launch cleanup failed; retained diagnostics at {directory}: {error}") from error

    async def close_session(self, session_id=None, timeout=5.0):
        target = self._target(session_id)
        try:
            session = await asyncio.to_thread(select_viewer_session, target, timeout=self.discovery_timeout)
            process = await asyncio.to_thread(_identity, session.pid)
            if process is not None and session.process_created_at is not None:
                if process.create_time() != session.process_created_at:
                    raise MCPViewerError("Viewer process identity changed; refusing to terminate a reused PID.")
            # Retain a creation-time-aware Process object before sending shutdown.
            if process is not None:
                process.create_time()
            request = urllib.request.Request(session.base_url + "/api/mcp/v1/shutdown", data=b"{}",
                headers={"Authorization": "Bearer " + session.token, "Content-Type": "application/json"}, method="POST")
            def shutdown():
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    response.read()
            try:
                await asyncio.to_thread(shutdown)
            except (OSError, urllib.error.URLError):
                pass  # The process can exit before its HTTP acknowledgement arrives.
            deadline = time.monotonic() + timeout
            while await asyncio.to_thread(_alive, process):
                if time.monotonic() >= deadline:
                    await asyncio.to_thread(_terminate_tree, process)
                    break
                await asyncio.sleep(0.1)
            if await asyncio.to_thread(_alive, process):
                raise MCPViewerError("Viewer is still running after shutdown.")
            if not await asyncio.to_thread(remove_viewer_session, session):
                raise MCPViewerError("Viewer exited, but its session descriptor could not be removed.")
            async with self._selection_lock:
                if self.connected_session_id == session.session_id:
                    self.connected_session_id = None
            return {"closed": True, "session_id": session.session_id, "pid": session.pid}
        except (LookupError, psutil.Error, OSError) as error:
            raise MCPViewerError(f"Could not close Viewer {target}: {error}") from error

    async def list_sessions(self):
        sessions = await asyncio.to_thread(discover_viewer_sessions, timeout=self.discovery_timeout)
        return {"sessions": [{"session_id": s.session_id, "pid": s.pid, "started_at": s.started_at} for s in sessions],
                "connected_session_id": self.connected_session_id,
                "automatic_selection": len(sessions) == 1}

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
                self._target(session_id),
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
