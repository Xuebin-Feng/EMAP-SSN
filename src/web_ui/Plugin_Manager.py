# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Discovery and registration for bundled Viewer web-utility plugins."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib
import os
import re
from types import ModuleType
from typing import Callable, Mapping


WEB_PLUGIN_API_VERSION = 1
_PLUGIN_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_MODULE_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_CALLABLE_RE = re.compile(r"^[A-Za-z_]\w*$")
_REQUIRED_MANIFEST_KEYS = frozenset(
    {"api_version", "id", "backend", "register", "activate"}
)
_RESERVED_ROUTE_PREFIXES = ("/api/", "/fonts/")


class WebPluginError(ValueError):
    """Raised when a bundled web plugin violates the registration contract."""


@dataclass(frozen=True)
class WebPluginDescriptor:
    api_version: int
    plugin_id: str
    backend: str
    register_callable: str
    activate_callable: str
    source_path: str


@dataclass(frozen=True)
class WebPluginDiagnostic:
    source_path: str
    plugin_id: str | None
    stage: str
    message: str

    def format(self) -> str:
        label = f"'{self.plugin_id}'" if self.plugin_id else "<unknown>"
        return (
            f"Web plugin {label} ({os.path.basename(self.source_path)}): "
            f"{self.stage} failed: {self.message}"
        )


def _literal_manifest(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError) as error:
        raise WebPluginError(str(error)) from error

    manifest_node = None
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == "WEB_PLUGIN"
            for target in targets
        ):
            manifest_node = node.value
            break

    if manifest_node is None:
        raise WebPluginError("missing literal WEB_PLUGIN manifest")
    try:
        manifest = ast.literal_eval(manifest_node)
    except (ValueError, TypeError, SyntaxError) as error:
        raise WebPluginError("WEB_PLUGIN must be a literal dictionary") from error
    if not isinstance(manifest, dict):
        raise WebPluginError("WEB_PLUGIN must be a literal dictionary")
    return manifest


def read_plugin_descriptor(path: str) -> WebPluginDescriptor:
    """Read and validate one descriptor without importing its backend."""
    manifest = _literal_manifest(path)
    if not all(isinstance(key, str) for key in manifest):
        raise WebPluginError("WEB_PLUGIN field names must be strings")
    keys = frozenset(manifest)
    missing = sorted(_REQUIRED_MANIFEST_KEYS - keys)
    unknown = sorted(keys - _REQUIRED_MANIFEST_KEYS)
    if missing:
        raise WebPluginError(f"missing manifest field(s): {', '.join(missing)}")
    if unknown:
        raise WebPluginError(f"unknown manifest field(s): {', '.join(unknown)}")

    api_version = manifest["api_version"]
    plugin_id = manifest["id"]
    backend = manifest["backend"]
    register_callable = manifest["register"]
    activate_callable = manifest["activate"]

    if isinstance(api_version, bool) or api_version != WEB_PLUGIN_API_VERSION:
        raise WebPluginError(
            f"unsupported API version {api_version!r}; expected {WEB_PLUGIN_API_VERSION}"
        )
    if not isinstance(plugin_id, str) or not _PLUGIN_ID_RE.fullmatch(plugin_id):
        raise WebPluginError(
            "id must start with a lowercase letter and contain only lowercase "
            "letters, digits, and underscores"
        )
    if not isinstance(backend, str) or not _MODULE_RE.fullmatch(backend):
        raise WebPluginError("backend must be a dotted Python module name")
    for field_name, value in (
        ("register", register_callable),
        ("activate", activate_callable),
    ):
        if not isinstance(value, str) or not _CALLABLE_RE.fullmatch(value):
            raise WebPluginError(f"{field_name} must be a Python callable name")

    return WebPluginDescriptor(
        api_version=api_version,
        plugin_id=plugin_id,
        backend=backend,
        register_callable=register_callable,
        activate_callable=activate_callable,
        source_path=os.path.abspath(path),
    )


def discover_plugin_descriptors(plugin_dir: str):
    """Return valid descriptors and diagnostics in deterministic filename order."""
    descriptors = []
    diagnostics = []
    seen_ids = set()
    if not os.path.isdir(plugin_dir):
        diagnostics.append(
            WebPluginDiagnostic(
                source_path=os.path.abspath(plugin_dir),
                plugin_id=None,
                stage="discovery",
                message="plugin directory does not exist",
            )
        )
        return descriptors, diagnostics

    for name in sorted(os.listdir(plugin_dir), key=str.casefold):
        if not name.endswith(".py") or name == "__init__.py":
            continue
        path = os.path.join(plugin_dir, name)
        descriptor = None
        try:
            descriptor = read_plugin_descriptor(path)
            if descriptor.plugin_id in seen_ids:
                raise WebPluginError(
                    f"duplicate plugin id '{descriptor.plugin_id}'"
                )
            seen_ids.add(descriptor.plugin_id)
            descriptors.append(descriptor)
        except Exception as error:
            diagnostics.append(
                WebPluginDiagnostic(
                    source_path=os.path.abspath(path),
                    plugin_id=descriptor.plugin_id if descriptor else None,
                    stage="manifest validation",
                    message=str(error),
                )
            )
    return descriptors, diagnostics


class WebPluginRegistry:
    """Per-Viewer routes, actions, state providers, and plugin ownership."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.actions: dict[str, Callable] = {}
        self.static_routes: dict[str, str] = {}
        self._action_owners: dict[str, str] = {}
        self._route_owners: dict[str, str] = {}
        self._state_providers: dict[tuple[str, str], Callable] = {}
        self.registered_plugins: set[str] = set()
        self._active_registration_owner: str | None = None

    def _validate_registration_owner(self, plugin_id: str):
        if (
            self._active_registration_owner is not None
            and plugin_id != self._active_registration_owner
        ):
            raise WebPluginError(
                f"plugin '{self._active_registration_owner}' attempted to "
                f"register a capability as plugin '{plugin_id}'"
            )

    @staticmethod
    def _normalize_route_prefix(prefix: str) -> str:
        if not isinstance(prefix, str) or not prefix.startswith("/"):
            raise WebPluginError("route prefix must begin with '/'")
        normalized = prefix if prefix.endswith("/") else prefix + "/"
        if normalized == "/" or ".." in normalized.split("/"):
            raise WebPluginError(f"invalid route prefix '{prefix}'")
        return normalized

    def register_action(self, plugin_id: str, action: str, handler: Callable):
        self._validate_registration_owner(plugin_id)
        if not isinstance(action, str) or not action.strip():
            raise WebPluginError("action name must be a non-empty string")
        if not callable(handler):
            raise WebPluginError(f"action '{action}' handler is not callable")
        owner = self._action_owners.get(action)
        if owner is not None and owner != plugin_id:
            raise WebPluginError(
                f"action '{action}' is already registered by plugin '{owner}'"
            )
        self.actions[action] = handler
        self._action_owners[action] = plugin_id

    def register_static_route(self, plugin_id: str, prefix: str, local_dir: str):
        self._validate_registration_owner(plugin_id)
        normalized = self._normalize_route_prefix(prefix)
        if any(
            normalized.startswith(reserved) or reserved.startswith(normalized)
            for reserved in _RESERVED_ROUTE_PREFIXES
        ):
            raise WebPluginError(f"route prefix '{normalized}' is reserved")
        for existing, owner in self._route_owners.items():
            if owner != plugin_id and (
                normalized.startswith(existing) or existing.startswith(normalized)
            ):
                raise WebPluginError(
                    f"route prefix '{normalized}' conflicts with '{existing}' "
                    f"from plugin '{owner}'"
                )
        self.static_routes[normalized] = os.path.abspath(local_dir)
        self._route_owners[normalized] = plugin_id

    def register_state_provider(
        self, plugin_id: str, name: str, provider: Callable
    ):
        self._validate_registration_owner(plugin_id)
        if not isinstance(name, str) or not name.strip():
            raise WebPluginError("state-provider name must be a non-empty string")
        if not callable(provider):
            raise WebPluginError(f"state provider '{name}' is not callable")
        self._state_providers[(plugin_id, name)] = provider

    def apply_state_providers(self, state: Mapping) -> dict:
        current = dict(state)
        for (plugin_id, name), provider in sorted(self._state_providers.items()):
            try:
                updated = provider(self.viewer, dict(current))
                if not isinstance(updated, Mapping):
                    raise TypeError("provider must return a mapping")
                current = dict(updated)
            except Exception as error:
                print(
                    f"Web plugin '{plugin_id}' state provider '{name}' failed: "
                    f"{error}"
                )
        return current

    def snapshot(self):
        return (
            dict(self.actions),
            dict(self.static_routes),
            dict(self._action_owners),
            dict(self._route_owners),
            dict(self._state_providers),
            set(self.registered_plugins),
        )

    def restore(self, snapshot):
        actions, routes, action_owners, route_owners, providers, registered = snapshot
        self.actions.clear()
        self.actions.update(actions)
        self.static_routes.clear()
        self.static_routes.update(routes)
        self._action_owners = action_owners
        self._route_owners = route_owners
        self._state_providers = providers
        self.registered_plugins = registered


def ensure_registry(viewer) -> WebPluginRegistry:
    registry = getattr(viewer, "web_plugin_registry", None)
    if registry is not None:
        return registry
    registry = WebPluginRegistry(viewer)
    legacy_actions = getattr(viewer, "web_action_handlers", {})
    for name, handler in legacy_actions.items():
        registry.register_action("legacy", name, handler)
    server = getattr(viewer, "web_server", None)
    if server is not None:
        for prefix, local_dir in getattr(server, "static_routes", {}).items():
            normalized = registry._normalize_route_prefix(prefix)
            registry.static_routes[normalized] = local_dir
            registry._route_owners[normalized] = "core"
        server.static_routes = registry.static_routes
    viewer.web_plugin_registry = registry
    viewer.web_action_handlers = registry.actions
    return registry


class WebPluginManager:
    """Discover, import, and atomically register bundled web plugins."""

    def __init__(self, viewer, plugin_dir: str | None = None):
        self.viewer = viewer
        self.registry = ensure_registry(viewer)
        self.plugin_dir = plugin_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "plugins"
        )
        self.descriptors: dict[str, WebPluginDescriptor] = {}
        self.modules: dict[str, ModuleType] = {}
        self.diagnostics: list[WebPluginDiagnostic] = []

    def _record_error(self, descriptor, stage, error):
        diagnostic = WebPluginDiagnostic(
            source_path=descriptor.source_path,
            plugin_id=descriptor.plugin_id,
            stage=stage,
            message=str(error),
        )
        self.diagnostics.append(diagnostic)
        print(diagnostic.format())

    def discover_and_register(self):
        descriptors, diagnostics = discover_plugin_descriptors(self.plugin_dir)
        self.diagnostics.extend(diagnostics)
        for diagnostic in diagnostics:
            print(diagnostic.format())
        for descriptor in descriptors:
            self.descriptors[descriptor.plugin_id] = descriptor
            self._register_descriptor(descriptor)
        return dict(self.descriptors)

    def _register_descriptor(self, descriptor: WebPluginDescriptor):
        if descriptor.plugin_id in self.registry.registered_plugins:
            return True
        try:
            module = importlib.import_module(descriptor.backend)
            register_callable = getattr(module, descriptor.register_callable)
            activate_callable = getattr(module, descriptor.activate_callable)
            if not callable(register_callable) or not callable(activate_callable):
                raise WebPluginError("declared registration callables are not callable")
        except Exception as error:
            self._record_error(descriptor, "backend import", error)
            return False

        snapshot = self.registry.snapshot()
        try:
            self.registry._active_registration_owner = descriptor.plugin_id
            register_callable(self.registry, self.viewer)
            self.registry.registered_plugins.add(descriptor.plugin_id)
            self.modules[descriptor.plugin_id] = module
            return True
        except Exception as error:
            self.registry.restore(snapshot)
            self._record_error(descriptor, "backend registration", error)
            return False
        finally:
            self.registry._active_registration_owner = None

    def activate(self, plugin_id: str):
        descriptor = self.descriptors.get(plugin_id)
        if descriptor is None:
            raise WebPluginError(f"unknown web plugin '{plugin_id}'")
        if plugin_id not in self.registry.registered_plugins:
            if not self._register_descriptor(descriptor):
                raise WebPluginError(f"web plugin '{plugin_id}' could not be registered")
        module = self.modules.get(plugin_id) or importlib.import_module(
            descriptor.backend
        )
        return getattr(module, descriptor.activate_callable)(self.viewer)
