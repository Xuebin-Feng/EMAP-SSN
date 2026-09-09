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
from utilities.Viewer_Sessions import (
    SESSION_DIRECTORY_ENV, LAUNCH_ID_ENV, session_directory,
    discover_viewer_sessions, remove_viewer_session, select_viewer_session,
    session_alias,
)
from desktop.Viewer_State import read_viewer_settings, validate_viewer_document, ViewerSettingsError


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
    def __init__(self, project_root=None, *, discovery_timeout=0.5, request_timeout=5.0, transport_closed=None):
        self.project_root = os.path.abspath(project_root or Path(__file__).resolve().parents[3])
        self.discovery_timeout = float(discovery_timeout)
        self.request_timeout = float(request_timeout)
        self.connected_session_id = None
        self._selection_lock = asyncio.Lock()
        self._monitor_task = None
        self._log_directories = {}
        self._transport_closed = transport_closed

    def _remember_session(self, session):
        if session.launch_id:
            try:
                launch_id = uuid.UUID(session.launch_id).hex
            except ValueError:
                pass
            else:
                self._log_directories[session.session_id] = Path(session_directory()) / "launches" / launch_id
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_connection())

    async def _monitor_connection(self):
        misses = 0
        previous = None
        while self.connected_session_id is not None:
            if self._transport_closed is not None and self._transport_closed():
                self.connected_session_id = None
                return
            target = self.connected_session_id
            if target != previous:
                misses = 0
                previous = target
            sessions = await asyncio.to_thread(discover_viewer_sessions, timeout=self.discovery_timeout)
            misses = 0 if any(s.session_id == target for s in sessions) else misses + 1
            if misses >= 3:
                async with self._selection_lock:
                    if self.connected_session_id == target:
                        self.connected_session_id = None
                        return
            await asyncio.sleep(1)

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
        self._remember_session(session)
        return {"connected": True, "session_id": session.session_id, "session_alias": session_alias(session.session_id), "pid": session.pid}

    async def disconnect_session(self):
        async with self._selection_lock:
            previous = self.connected_session_id
            self.connected_session_id = None
        task, self._monitor_task = self._monitor_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
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
        script = Path(self.project_root) / "src" / "EMAPSSN_Config.py"
        command = [sys.executable, "-u", str(script), "--headless", "launch-viewer",
                   "--settings", str(snapshot), "--delete-settings", "--viewer-mode", mode]
        if mode == "normal":
            command = [sys.executable, "-u", str(Path(__file__).resolve().parent / "Viewer_Terminal.py"),
                       str(directory), *command]
        env = os.environ.copy()
        for key in ("SSN_VIEWER_SETTINGS_PATH", "SSN_TARGET_CACHE_PATH", "SSN_TARGET_CACHE_MODE", "SSN_TARGET_CACHE",
                    "SSN_VIEWER_HEADLESS", "SSN_VIEWER_EXPLICIT_SETTINGS"):
            env.pop(key, None)
        env[SESSION_DIRECTORY_ENV] = session_directory()
        env[LAUNCH_ID_ENV] = launch_id
        if mode == "headless":
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
                child_options = ("stdin=None,stdout=None,stderr=None,creationflags=subprocess.CREATE_NEW_CONSOLE"
                                 if mode == "normal" else
                                 "stdin=subprocess.DEVNULL,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP")
                broker = (
                    "import subprocess,sys,json,psutil; "
                    f"p=subprocess.Popen(json.loads(sys.argv[1]),{child_options}); "
                    "open(sys.argv[2],'w').write(json.dumps({'pid':p.pid,'created':psutil.Process(p.pid).create_time()}))"
                )
                command = [sys.executable, "-c", broker, json.dumps(command), str(identity_path)]
            with stdout_path.open("ab", buffering=0) as out, stderr_path.open("ab", buffering=0) as err:
                if mode == "normal" and sys.platform != "win32":
                    from utilities.Terminal_Launcher import launch_in_terminal
                    proc = launch_in_terminal(command, cwd=self.project_root, env=env)
                else:
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
                    self._remember_session(session)
                    ready = True
                    # A daemon reaps the launcher without owning the independent Viewer lifetime.
                    threading.Thread(target=proc.wait, daemon=True, name="viewer-launcher-reaper").start()
                    return {"status": "ready", "session_id": session.session_id, "pid": session.pid,
                            "session_alias": session_alias(session.session_id),
                            "base_url": session.base_url, "started_at": session.started_at, "mode": mode,
                            "cache_path": settings["inputs"]["TARGET_CACHE_PATH"],
                            "stdout_log": str(stdout_path), "stderr_log": str(stderr_path)}
                if (sys.platform == "win32" or mode == "headless") and not await asyncio.to_thread(_alive, root_process):
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
                    terminal_identity = directory / "terminal-process.json"
                    if terminal_identity.exists():
                        recorded = json.loads(terminal_identity.read_text())
                        terminal = _identity(recorded["pid"])
                        if terminal is not None and terminal.create_time() == recorded["created"]:
                            await asyncio.shield(asyncio.to_thread(_terminate_tree, terminal))
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

    async def list_sessions(self, offset=0, limit=25, max_bytes=16384):
        from desktop.Viewer_Inspection import encoded
        if offset < 0 or not 1 <= limit <= 100 or not 1024 <= max_bytes <= 65536:
            raise MCPViewerError("Invalid session page bounds")
        target = self.connected_session_id
        sessions = sorted(await asyncio.to_thread(discover_viewer_sessions, timeout=self.discovery_timeout), key=lambda s: s.session_id)
        async with self._selection_lock:
            if target == self.connected_session_id and target not in {s.session_id for s in sessions}:
                self.connected_session_id = None
        result = {"sessions": [], "connected_session_id": self.connected_session_id,
                  "automatic_selection": len(sessions) == 1, "total": len(sessions),
                  "offset": offset, "next_offset": offset, "complete": False}
        for session in sessions[offset:offset + limit]:
            item = {"session_id": session.session_id, "session_alias": session_alias(session.session_id),
                    "pid": session.pid, "started_at": session.started_at}
            try:
                info = await asyncio.to_thread(self._request, session, "/api/mcp/v1/session")
                item["inputs"] = info.get("inputs", {})
                item["inspection_capabilities"] = info.get("inspection_capabilities", [])
            except (MCPViewerError, OSError, ValueError, TypeError) as error:
                item["metadata_error"] = str(error)[:256]
            result["sessions"].append(item)
            if len(encoded(result)) > max_bytes - 64:
                result["sessions"].pop()
                if not result["sessions"]:
                    raise MCPViewerError("Session identity exceeds byte budget; increase max_bytes")
                break
            result["next_offset"] += 1
        result["complete"] = result["next_offset"] >= len(sessions)
        result["returned_count"] = len(result["sessions"])
        return result

    async def command_action(self, action, arguments, session_id=None):
        target = self._target(session_id)
        try:
            session = await asyncio.to_thread(select_viewer_session, target, timeout=self.discovery_timeout)
        except LookupError as error:
            raise MCPViewerError(str(error)) from error
        info = await asyncio.to_thread(self._request, session, '/api/mcp/v1/session')
        if 'commands_v1' not in info.get('inspection_capabilities', []):
            raise MCPViewerError('Viewer does not support the command portal; upgrade and restart the Viewer.')
        return await asyncio.to_thread(self._request, session, '/api/mcp/v1/commands', {'action': action, 'arguments': arguments})

    async def inspect_data(self, action, arguments, session_id=None):
        target = self._target(session_id)
        try:
            session = await asyncio.to_thread(select_viewer_session, target, timeout=self.discovery_timeout)
        except LookupError as error:
            async with self._selection_lock:
                if target == self.connected_session_id:
                    self.connected_session_id = None
            raise MCPViewerError(str(error)) from error
        capabilities = await asyncio.to_thread(self._request, session, "/api/mcp/v1/session")
        if "snapshots_v1" not in capabilities.get("inspection_capabilities", []):
            raise MCPViewerError("Viewer does not support snapshots; upgrade and restart the Viewer.")
        required = []
        if arguments.get("include_alignment") or action == "get_residue_distribution":
            required.append("alignment_snapshots_v1")
        if arguments.get("fields") is not None:
            required.append("node_projection_v1")
        if any(cap not in capabilities.get("inspection_capabilities", []) for cap in required):
            raise MCPViewerError("Viewer lacks requested scientific inspection capabilities; upgrade and restart the Viewer.")
        return await asyncio.to_thread(self._request, session, "/api/mcp/v1/data", {"action": action, "arguments": arguments})

    async def get_summary(self, session_id=None, max_bytes=16384):
        return await self.inspect_data("get_summary", {"max_bytes": max_bytes}, session_id)

    async def read_log(self, session_id=None, *, stream="stdout", offset=0, limit=8192):
        if stream not in {"stdout", "stderr"} or offset < 0 or not 1 <= limit <= 1048576:
            raise MCPViewerError("Expected stdout/stderr, nonnegative byte offset, and limit 1..1048576.")
        target = self._target(session_id)
        directory = self._log_directories.get(target)
        if directory is None:
            try:
                session = await asyncio.to_thread(select_viewer_session, target, timeout=self.discovery_timeout)
                if not session.launch_id:
                    raise MCPViewerError("This Viewer was not launched with MCP output capture. For MCP-submitted command output, use read_command_output with request_id or submission_id.")
                launch_id = uuid.UUID(session.launch_id).hex
                directory = Path(session_directory()) / "launches" / launch_id
                self._log_directories[session.session_id] = directory
                target = session.session_id
            except (LookupError, ValueError) as error:
                raise MCPViewerError(str(error)) from error
        def read():
            try:
                with (directory / f"{stream}.log").open("rb") as handle:
                    size = os.fstat(handle.fileno()).st_size
                    handle.seek(min(offset, size))
                    data = handle.read(limit)
                    next_offset = handle.tell()
            except OSError as error:
                raise MCPViewerError(f"Could not read Viewer output: {error}") from error
            return {"session_id": target, "stream": stream, "offset": min(offset, size),
                    "next_offset": next_offset, "size": size, "eof": next_offset >= size,
                    "text": data.decode("utf-8", errors="replace")}
        return await asyncio.to_thread(read)

    async def query_nodes(self, snapshot_id, session_id=None, **arguments):
        return await self.inspect_data("query_nodes", dict(snapshot_id=snapshot_id, **arguments), session_id)

    async def _get(self, session_id, endpoint):
        target = self._target(session_id)
        try:
            session = await asyncio.to_thread(
                select_viewer_session,
                target,
                timeout=self.discovery_timeout,
            )
        except LookupError as error:
            async with self._selection_lock:
                if target == self.connected_session_id:
                    self.connected_session_id = None
            raise MCPViewerError(str(error)) from error
        return await asyncio.to_thread(self._request, session, endpoint)

    def _request(self, session, endpoint, data=None):
        request = urllib.request.Request(
            f"{session.base_url}{endpoint}",
            headers={"Authorization": f"Bearer {session.token}", "Content-Type": "application/json"},
            data=json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None,
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


ViewerClient = MCPViewerClient
ViewerError = MCPViewerError

__all__ = ["MCPViewerClient", "MCPViewerError", "ViewerClient", "ViewerError"]

