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


# One start_session or wait_session call waits this long before handing a
# still-loading Viewer back to the caller; Claude's desktop app may end a single
# MCP tool call after about a minute.
DEFAULT_READY_TIMEOUT = 45.0


def _launch_key(launch_id):
    """Return the canonical form of a launch ID, which also names its directory."""
    try:
        return uuid.UUID(str(launch_id)).hex
    except ValueError:
        raise MCPViewerError(
            f"launch_id: expected the ID returned by start_session, not {launch_id!r}."
        ) from None


def _read_launch(directory):
    try:
        launch = json.loads((directory / "launch.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise MCPViewerError(
            f"No MCP launch {directory.name} exists. For running Viewers use "
            "emapssn_viewer_data(action='list_sessions')."
        ) from None
    except (OSError, ValueError) as error:
        raise MCPViewerError(f"Cannot read launch record {directory}: {error}") from error
    if (not isinstance(launch, dict) or launch.get("launch_id") != directory.name
            or launch.get("mode") not in {"normal", "headless"}
            or isinstance(launch.get("started_epoch"), bool)
            or not isinstance(launch.get("started_epoch"), (int, float))
            or not isinstance(launch.get("cache_path"), str)):
        raise MCPViewerError(f"Invalid launch record at {directory}.")
    return launch


def _recorded_process(path):
    """Return the process an identity file records while it still runs, else None."""
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
        process = _identity(int(recorded["pid"]))
        if process is not None and process.create_time() == recorded["created"] and _alive(process):
            return process
    except (OSError, ValueError, KeyError, TypeError, psutil.Error):
        pass
    return None


# Viewer_Terminal.py records itself before anything else, so a normal POSIX
# launch whose terminal command has exited and still has not run it this long
# after the launch started never will.
TERMINAL_START_GRACE = 60.0


def _terminal_alive(directory, started_epoch, terminal=None):
    """Whether a normal POSIX launch can still become ready.

    Once the terminal wrapper records itself, the launch lives until that
    process exits. Before then the terminal command (``terminal``, when this
    process started it, else the process ``process.json`` records) may still
    start the wrapper; macOS's osascript waits there while the user is asked to
    allow control of Terminal. A command that failed, as an emulator without a
    display or osascript refused that control does, raises MCPViewerError. One
    that exited cleanly, as emulators handing their window to a running server
    do, leaves TERMINAL_START_GRACE seconds from the launch's start.
    """
    if terminal is not None:
        code = terminal.poll()
    else:
        code = None if _recorded_process(directory / "process.json") is not None else 0
    # Checked after the command, so a wrapper that recorded itself as the
    # command exited is not mistaken for one that never started.
    identity = directory / "terminal-process.json"
    if identity.exists():
        return _recorded_process(identity) is not None
    if code is None:
        return True
    if code != 0:
        raise MCPViewerError(
            f"The terminal command exited with code {code} before starting the Viewer. "
            "Normal mode needs a desktop session (on macOS, permission to control "
            "Terminal); use headless mode without one."
        )
    if time.time() - started_epoch < TERMINAL_START_GRACE:
        return True
    raise MCPViewerError(
        f"No terminal started the Viewer within {TERMINAL_START_GRACE:.0f} s of the launch. "
        "Normal mode needs a desktop session with a working terminal emulator; "
        "use headless mode without one."
    )


def _launch_progress(directory):
    """Return the last line a launch printed and its age in seconds."""
    path = directory / "stdout.log"
    try:
        status = path.stat()
        with path.open("rb") as handle:
            handle.seek(max(0, status.st_size - 4096))
            lines = [line.strip() for line in handle.read().decode("utf-8", errors="replace").splitlines()]
    except OSError:
        return None, None
    lines = [line for line in lines if line]
    if not lines:
        return None, None
    return lines[-1][:300], round(max(0.0, time.time() - status.st_mtime), 1)


def _tree_cpu_seconds(process):
    """CPU time used by a launch's process tree; growth between calls means work."""
    if process is None:
        return None
    total = 0.0
    try:
        targets = [process, *process.children(recursive=True)]
    except psutil.Error:
        return None
    for target in targets:
        try:
            times = target.cpu_times()
        except psutil.Error:
            continue
        total += times.user + times.system
    return round(total, 1)


def _log_tails(directory):
    tails = []
    for path in (directory / "stdout.log", directory / "stderr.log"):
        if path.exists():
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 4096))
                tails.append(handle.read().decode("utf-8", errors="replace"))
    return "\n".join(tails)


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
            raise MCPViewerError(
                "No Viewer is connected. Call emapssn_viewer_control(action='connect_session') "
                "or supply session_id."
            )
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

    async def launch_session(self, *, settings_document=None, settings_path=None, mode="normal",
                             timeout=DEFAULT_READY_TIMEOUT):
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
        # The record lets wait_session and close_session reach a launch that
        # outlives this call, including from a restarted backend.
        launch = {"launch_id": launch_id, "mode": mode, "started_epoch": time.time(),
                  "cache_path": settings["inputs"]["TARGET_CACHE_PATH"]}
        (directory / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
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
        # CREATE_NO_WINDOW rather than DETACHED_PROCESS: a venv's python.exe is a
        # redirector that starts the real interpreter as its child, and a
        # redirector without any console makes Windows allocate a new, visible
        # console for that child. A hidden console is inherited instead.
        options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB}
                   if sys.platform == "win32" else {"start_new_session": True})
        proc = None
        root_process = None
        ready = False
        keep_running = False
        try:
            # Windows MCP transports terminate their subprocess trees on disconnect.
            # A short-lived broker exits before readiness, severing that ownership
            # chain while recording the independent launcher's exact identity.
            identity_path = directory / "process.json"
            if sys.platform == "win32":
                child_options = ("stdin=None,stdout=None,stderr=None,creationflags=subprocess.CREATE_NEW_CONSOLE"
                                 if mode == "normal" else
                                 "stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP")
                broker = (
                    "import subprocess,sys,json,psutil; "
                    f"p=subprocess.Popen(json.loads(sys.argv[1]),{child_options}); "
                    "open(sys.argv[2],'w').write(json.dumps({'pid':p.pid,'created':psutil.Process(p.pid).create_time()}))"
                )
                command = [sys.executable, "-c", broker, json.dumps(command), str(identity_path)]
            with stdout_path.open("ab", buffering=0) as out, stderr_path.open("ab", buffering=0) as err:
                if mode == "normal" and sys.platform != "win32":
                    from utilities.Terminal_Launcher import launch_in_terminal
                    # The terminal command's own output, such as an emulator's
                    # "cannot open display" or osascript refused control of
                    # Terminal, belongs in the launch log. This server's stdin and
                    # stdout carry the MCP protocol, so the command gets neither.
                    proc = launch_in_terminal(command, cwd=self.project_root, env=env,
                                              stdin=subprocess.DEVNULL, stdout=err, stderr=err)
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
                    raise MCPViewerError("Independent Viewer launcher exited before readiness; "
                                         f"last output: {_launch_progress(directory)[0]!r}.")
            elif root_process is not None:
                # The headless Viewer launcher, or the terminal command that starts
                # a normal one; wait_session and close_session reach it through this.
                identity_path.write_text(json.dumps({"pid": proc.pid, "created": root_process.create_time()}))
            if sys.platform == "win32" or mode == "headless":
                owner = root_process
                def alive():
                    return _alive(root_process)
            else:
                owner = None  # root_process is the terminal emulator, not the Viewer.
                def alive():
                    return _terminal_alive(directory, launch["started_epoch"], terminal=proc)
            # Only a POSIX headless Popen is the Viewer itself; on Windows it is the broker.
            child = proc if mode == "headless" and sys.platform != "win32" else None
            result = await self._await_launch(directory, launch, alive, timeout, owner=owner, child=child)
            ready = result["status"] == "ready"
            keep_running = not ready
            return result
        except asyncio.CancelledError:
            # A cancelled call, including an MCP client's own tool timeout, leaves a
            # spawned Viewer loading; wait_session or close_session can still reach it.
            keep_running = proc is not None
            raise
        except Exception as error:
            if isinstance(error, PermissionError) and sys.platform == "win32":
                error = MCPViewerError(
                    "The Windows host denied an independent Viewer process (its Job Object may forbid breakaway). "
                    "Open the Viewer through the GUI or CLI, then use "
                    "emapssn_viewer_control(action='connect_session')."
                )
            note = " Remaining launch processes were terminated." if proc is not None else ""
            raise MCPViewerError(f"{error}{note} Logs: {directory}\n" + _log_tails(directory)) from error
        finally:
            if ready or keep_running:
                if proc is not None and proc.returncode is None:
                    # A daemon reaps the launcher without owning the independent Viewer lifetime.
                    threading.Thread(target=proc.wait, daemon=True, name="viewer-launcher-reaper").start()
            else:
                try:
                    await self._terminate_launch(directory, launch_id, extra=root_process)
                    if proc is not None:
                        await asyncio.to_thread(proc.wait, timeout=5)
                except Exception as error:
                    raise MCPViewerError(f"Launch cleanup failed; retained diagnostics at {directory}: {error}") from error

    async def _await_launch(self, directory, launch, alive, timeout, *, owner=None, child=None):
        """Wait until a launch answers, exits, or ``timeout`` passes; never stop it."""
        launch_id = launch["launch_id"]
        deadline = time.monotonic() + timeout
        probe_error = None
        while True:
            sessions = await asyncio.to_thread(discover_viewer_sessions, timeout=self.discovery_timeout)
            session = next((item for item in sessions if item.launch_id == launch_id), None)
            if session is not None:
                try:
                    await asyncio.to_thread(self._request, session, "/api/mcp/v1/summary")
                except MCPViewerError as error:
                    # Published, but its Qt thread has not answered yet (for example
                    # during the first paint); keep polling a live Viewer.
                    probe_error = str(error)
                else:
                    async with self._selection_lock:
                        self.connected_session_id = session.session_id
                    self._remember_session(session)
                    return {"status": "ready", "session_id": session.session_id, "pid": session.pid,
                            "session_alias": session_alias(session.session_id),
                            "base_url": session.base_url, "started_at": session.started_at,
                            "mode": launch["mode"], "cache_path": launch["cache_path"], "launch_id": launch_id,
                            "stdout_log": str(directory / "stdout.log"),
                            "stderr_log": str(directory / "stderr.log")}
            elif not await asyncio.to_thread(alive):
                code = child.poll() if child is not None else None
                phase, _ = _launch_progress(directory)
                raise MCPViewerError(
                    "Viewer exited before readiness"
                    + (f" (code {code})" if code is not None else "")
                    + f" after {time.time() - launch['started_epoch']:.1f} s; last output: {phase!r}."
                )
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.1)
        if owner is None:
            owner = await asyncio.to_thread(_recorded_process, directory / "terminal-process.json")
        phase, age = _launch_progress(directory)
        result = {
            "status": "starting", "launch_id": launch_id, "mode": launch["mode"],
            "elapsed_seconds": round(time.time() - launch["started_epoch"], 1), "ready_timeout": timeout,
            "phase": phase, "last_output_age_seconds": age,
            "launcher_pid": owner.pid if owner is not None else None,
            "cpu_seconds": await asyncio.to_thread(_tree_cpu_seconds, owner),
            "cache_path": launch["cache_path"],
            "stdout_log": str(directory / "stdout.log"), "stderr_log": str(directory / "stderr.log"),
            "message": ("The Viewer is still loading and was left running. Continue with wait_session and "
                        "this launch_id instead of starting another Viewer; close_session with launch_id "
                        "stops it."),
            "next_step": {"tool": "emapssn_viewer_control", "action": "wait_session",
                          "arguments": {"launch_id": launch_id}},
        }
        if probe_error is not None:
            result["readiness_probe_error"] = probe_error
        return result

    async def _terminate_launch(self, directory, launch_id, *, extra=None):
        """Stop the processes a launch recorded and any session it published."""
        stopped = []
        for name in ("terminal-process.json", "process.json"):
            process = await asyncio.to_thread(_recorded_process, directory / name)
            if process is not None:
                await asyncio.shield(asyncio.to_thread(_terminate_tree, process))
                stopped.append(process.pid)
        if extra is not None and await asyncio.to_thread(_alive, extra):
            await asyncio.shield(asyncio.to_thread(_terminate_tree, extra))
            stopped.append(extra.pid)
        for session in await asyncio.to_thread(discover_viewer_sessions, timeout=self.discovery_timeout):
            if session.launch_id == launch_id:
                actual = _identity(session.pid)
                if actual is not None and actual.create_time() == session.process_created_at:
                    await asyncio.shield(asyncio.to_thread(_terminate_tree, actual))
                    stopped.append(actual.pid)
                remove_viewer_session(session)
        (directory / "settings.json").unlink(missing_ok=True)
        return stopped

    async def wait_for_launch(self, launch_id, timeout=DEFAULT_READY_TIMEOUT):
        """Keep waiting for a handed-off launch; connect when ready, never stop it."""
        directory = Path(session_directory()) / "launches" / _launch_key(launch_id)
        launch = _read_launch(directory)
        owner = None
        if sys.platform == "win32" or launch["mode"] == "headless":
            owner = await asyncio.to_thread(_recorded_process, directory / "process.json")
            def alive():
                return owner is not None and _alive(owner)
        else:
            def alive():
                return _terminal_alive(directory, launch["started_epoch"])
        # An already-ready Viewer is found by its launch_id even after the launcher exits.
        try:
            return await self._await_launch(directory, launch, alive, timeout, owner=owner)
        except MCPViewerError as error:
            (directory / "settings.json").unlink(missing_ok=True)
            raise MCPViewerError(f"{error} Logs: {directory}\n" + _log_tails(directory)) from error

    async def close_session(self, session_id=None, timeout=5.0, *, launch_id=None):
        if launch_id is not None:
            if session_id is not None:
                raise MCPViewerError("Supply session_id or launch_id, not both.")
            return await self._close_launch(launch_id, timeout)
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

    async def _close_launch(self, launch_id, timeout):
        """Close a launch's Viewer, or stop it when it has not published a session yet."""
        directory = Path(session_directory()) / "launches" / _launch_key(launch_id)
        launch = _read_launch(directory)
        sessions = await asyncio.to_thread(discover_viewer_sessions, timeout=self.discovery_timeout)
        session = next((item for item in sessions if item.launch_id == launch["launch_id"]), None)
        if session is not None:
            return {**await self.close_session(session.session_id, timeout), "launch_id": launch["launch_id"]}
        try:
            stopped = await self._terminate_launch(directory, launch["launch_id"])
        except (psutil.Error, OSError) as error:
            raise MCPViewerError(f"Could not stop launch {launch['launch_id']}: {error}") from error
        if not stopped:
            raise MCPViewerError(f"No running Viewer belongs to launch {launch['launch_id']}.")
        return {"closed": True, "session_id": None, "pid": stopped[0], "launch_id": launch["launch_id"]}

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

