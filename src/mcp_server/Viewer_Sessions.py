# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Publish and discover authenticated local EMAP-SSN Viewer sessions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
import psutil


SESSION_PROTOCOL_VERSION = 1
LOOPBACK_HOST = "127.0.0.1"
SESSION_DIRECTORY_ENV = "SSN_VIEWER_SESSION_DIR"
LAUNCH_ID_ENV = "SSN_VIEWER_LAUNCH_ID"


def session_alias(session_id):
    """A human-readable UUID prefix; ambiguous prefixes are never auto-selected."""
    try:
        return uuid.UUID(str(session_id)).hex[:8].upper()
    except ValueError:
        return None  # Older/test descriptors may use non-UUID identifiers.


def ensure_viewer_identity(viewer):
    """Own identity on the Viewer so server restarts preserve it."""
    if not getattr(viewer, "inspection_session_id", None):
        viewer.inspection_session_id = str(uuid.uuid4())
    viewer.inspection_session_alias = session_alias(viewer.inspection_session_id)
    return viewer.inspection_session_id


@dataclass(frozen=True)
class ViewerSessionDescriptor:
    protocol_version: int
    session_id: str
    pid: int
    host: str
    port: int
    token: str
    started_at: str
    descriptor_path: str = ""
    launch_id: str | None = None
    process_created_at: float | None = None

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"


def session_directory():
    configured = os.environ.get(SESSION_DIRECTORY_ENV)
    if configured:
        return os.path.abspath(os.fspath(configured))
    base = os.path.join(
        tempfile.gettempdir(),
        "sequence_similarity_network_viewer",
        "viewer_sessions",
    )
    try:
        os.makedirs(base, exist_ok=True)
        os.listdir(base)
    except (OSError, PermissionError):
        base = os.path.join(
            tempfile.gettempdir(),
            "sequence_similarity_network_viewer_sessions",
        )
        os.makedirs(base, exist_ok=True)
    return base


def _secure_directory(path):
    os.makedirs(path, mode=0o700, exist_ok=True)
    if sys.platform != "win32":
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass


def publish_viewer_session(
    *,
    session_id,
    pid,
    port,
    token,
    host=LOOPBACK_HOST,
    started_at=None,
):
    """Atomically publish one user-local Viewer descriptor."""
    if host != LOOPBACK_HOST:
        raise ValueError("Viewer discovery may only publish the IPv4 loopback host.")
    descriptor_root = session_directory()
    _secure_directory(descriptor_root)
    descriptor_path = os.path.join(descriptor_root, f"viewer-{session_id}.json")
    descriptor = ViewerSessionDescriptor(
        protocol_version=SESSION_PROTOCOL_VERSION,
        session_id=str(session_id),
        pid=int(pid),
        host=host,
        port=int(port),
        token=str(token),
        started_at=(
            started_at
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        ),
        descriptor_path=descriptor_path,
        launch_id=os.environ.get(LAUNCH_ID_ENV),
        process_created_at=psutil.Process(pid).create_time(),
    )
    payload = asdict(descriptor)
    payload.pop("descriptor_path", None)
    partial_path = f"{descriptor_path}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    try:
        with open(partial_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        if sys.platform != "win32":
            try:
                os.chmod(partial_path, 0o600)
            except OSError:
                pass
        os.replace(partial_path, descriptor_path)
    finally:
        if os.path.exists(partial_path):
            os.unlink(partial_path)
    return descriptor


def remove_viewer_session(descriptor_or_path):
    """Remove a descriptor owned by a Viewer; repeated calls are safe."""
    path = (
        descriptor_or_path.descriptor_path
        if isinstance(descriptor_or_path, ViewerSessionDescriptor)
        else os.fspath(descriptor_or_path)
    )
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        return False
    return True


def _load_descriptor(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        descriptor = ViewerSessionDescriptor(
            protocol_version=int(payload["protocol_version"]),
            session_id=str(payload["session_id"]),
            pid=int(payload["pid"]),
            host=str(payload["host"]),
            port=int(payload["port"]),
            token=str(payload["token"]),
            started_at=str(payload["started_at"]),
            descriptor_path=os.path.abspath(path),
            launch_id=payload.get("launch_id"),
            process_created_at=(float(payload["process_created_at"])
                                if payload.get("process_created_at") is not None else None),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    if (
        descriptor.protocol_version != SESSION_PROTOCOL_VERSION
        or descriptor.host != LOOPBACK_HOST
        or not descriptor.session_id
        or descriptor.pid <= 0
        or not (1 <= descriptor.port <= 65535)
        or not descriptor.token
    ):
        return None
    return descriptor


def _process_is_running(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError):
        return None


def _validate_live_session(descriptor, timeout):
    request = urllib.request.Request(
        f"{descriptor.base_url}/api/mcp/v1/session",
        headers={"Authorization": f"Bearer {descriptor.token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (
        OSError,
        UnicodeError,
        ValueError,
        urllib.error.URLError,
    ):
        return False
    try:
        response_pid = int(payload.get("pid", -1))
    except (TypeError, ValueError):
        return False
    return (
        payload.get("protocol_version") == descriptor.protocol_version
        and payload.get("session_id") == descriptor.session_id
        and response_pid == descriptor.pid
        and (descriptor.launch_id is None or payload.get("launch_id") == descriptor.launch_id)
    )


def discover_viewer_sessions(*, timeout=0.5, prune_stale=True):
    """Return all authenticated live Viewer sessions in stable order."""
    root = session_directory()
    try:
        names = sorted(os.listdir(root))
    except (FileNotFoundError, PermissionError, OSError):
        return []
    sessions = []
    for name in names:
        if not name.startswith("viewer-") or not name.endswith(".json"):
            continue
        path = os.path.join(root, name)
        descriptor = _load_descriptor(path)
        if descriptor is None:
            continue
        if _process_is_running(descriptor.pid) is False:
            if prune_stale:
                remove_viewer_session(path)
            continue
        if _validate_live_session(descriptor, timeout):
            sessions.append(descriptor)
        elif prune_stale and _process_is_running(descriptor.pid) is False:
            remove_viewer_session(path)
    return sorted(sessions, key=lambda item: (item.started_at, item.session_id))


def select_viewer_session(session_id=None, *, timeout=0.5):
    """Select explicitly, or automatically only when exactly one Viewer is live."""
    sessions = discover_viewer_sessions(timeout=timeout)
    if session_id is not None:
        for session in sessions:
            if session.session_id.lower() == session_id.lower():
                return session
        matches = [s for s in sessions if session_alias(s.session_id) == session_id.upper()]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise LookupError("Ambiguous Viewer alias; use a full session ID: " + ", ".join(s.session_id for s in matches))
        raise LookupError(f"No live Viewer session has ID '{session_id}'.")
    if len(sessions) == 1:
        return sessions[0]
    if not sessions:
        raise LookupError("No live Viewer sessions were found.")
    raise LookupError(
        "Multiple Viewer sessions are running; select one by session ID."
    )


__all__ = [
    "LOOPBACK_HOST",
    "SESSION_DIRECTORY_ENV",
    "SESSION_PROTOCOL_VERSION",
    "ViewerSessionDescriptor",
    "discover_viewer_sessions",
    "publish_viewer_session",
    "remove_viewer_session",
    "select_viewer_session",
    "session_directory",
]
