# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
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

"""Discovery and registration for bundled Viewer web-utility plugins.

A plugin declares itself in a literal ``WEB_PLUGIN`` manifest
(src/web_ui/plugins/<id>.py, docs/web_plugin_development.md). Besides the
backend to register, the manifest describes the browser pages the plugin
serves: where each is, what it is called, which Studio theme and
Content-Security-Policy it takes, and which files hold its translated texts.
The web server, the translations and the Viewer's page opener read all of that
from here, so none of them names a page. Nothing in this module imports Qt, so
tools that run without the application (the translation update) can read the
manifests.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import glob
import importlib
import os
import re
from typing import Callable, Mapping

from utilities.Localization import MESSAGE_CONTEXT, Message


WEB_PLUGIN_API_VERSION = 1
_PLUGIN_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_MODULE_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_CALLABLE_RE = re.compile(r"^[A-Za-z_]\w*$")
_REQUIRED_MANIFEST_KEYS = frozenset({"api_version", "id", "backend", "register"})
_OPTIONAL_MANIFEST_KEYS = frozenset({"pages"})
_RESERVED_ROUTE_PREFIXES = ("/api/", "/fonts/")

# The names after /api/ that the web server answers itself (the event stream,
# the page actions and the MCP bridge). A plugin of such a name may not register
# an API route, whose prefix would be the core route's.
CORE_API_NAMES = frozenset({"action", "events", "mcp"})
API_METHODS = frozenset({"GET", "POST"})

# The only relaxations of the base Content-Security-Policy (Web_Server.PAGE_POLICY)
# a page may ask for, by directive: the sources it may add. The list is the
# server's decision, not a page's: a manifest that asks for anything else is
# refused. 'unsafe-eval' lets Mol*'s bundle compile code while it loads, and
# https: and data: let it reach public servers and take its encoder from a
# data: URL.
ALLOWED_CSP_RELAXATIONS = {
    "script-src": ("'unsafe-eval'",),
    "connect-src": ("https:", "data:"),
}

_PAGE_REQUIRED_KEYS = frozenset({"path", "label", "client"})
_PAGE_OPTIONAL_KEYS = frozenset({"themed", "csp", "i18n"})
_I18N_KEYS = frozenset({"context", "files"})
# A page's URL: a file of .html in the folder the web layer serves, or in one
# of its plugin's own static routes. Every segment starts with a letter,
# digit or underscore, so none is . or ..
_PAGE_PATH_RE = re.compile(r"^/[A-Za-z0-9_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)*\.html$")
# The label of an SSE client, as the server accepts one (Web_Server.event_client_from_path).
_CLIENT_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
_CONTEXT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
# A file of a page's texts, relative to src, which may name several by * and ?.
_I18N_FILE_RE = re.compile(r"^[A-Za-z0-9_*?][A-Za-z0-9_.*?-]*(?:/[A-Za-z0-9_*?][A-Za-z0-9_.*?-]*)*$")
_API_PATH_CHARACTERS = re.compile(r"^[A-Za-z0-9_./-]+$")


class WebPluginError(ValueError):
    """Raised when a bundled web plugin violates the registration contract."""


class WebPageUnavailable(LookupError):
    """Raised when a page no plugin registered is asked for."""


class PageLabel(Message):
    """The Message that names a plugin's page in the Viewer's lines and dialogs.

    Its English template is the label of the page's manifest, which the
    translation update lists in the catalog as it lists a Message(...) written
    in code (resources/languages/Update_Translations.py), so the label is
    translated as any other Message.
    """

    __slots__ = ()


@dataclass(frozen=True)
class WebPageDescriptor:
    """One browser page of a plugin, as its manifest declares it.

    path is the page's URL; label names it for people (a manifest's English,
    translated when it is shown); client is the id the page's event stream
    connects with, which tells whether the page is open. themed pages end
    their head with the Studio theme's colours. csp holds the relaxations
    of the base Content-Security-Policy the page needs, as ((directive,
    (source, ...)), ...), each of them on ALLOWED_CSP_RELAXATIONS. i18n_context
    is the catalog context of the page's texts and i18n_files the files, relative
    to src, that hold them (patterns, so *.js names several). Any value that is
    not allowed raises WebPluginError, whichever way the descriptor is made.
    """

    plugin_id: str
    path: str
    label: str
    client: str
    themed: bool = False
    csp: tuple = ()
    i18n_context: str | None = None
    i18n_files: tuple = ()

    def __post_init__(self):
        where = f"page '{self.path}'"
        if not isinstance(self.plugin_id, str) or not _PLUGIN_ID_RE.fullmatch(self.plugin_id):
            raise WebPluginError(f"{where}: the plugin id is not valid")
        if (not isinstance(self.path, str) or not _PAGE_PATH_RE.fullmatch(self.path)
                or any(self.path.startswith(prefix) for prefix in _RESERVED_ROUTE_PREFIXES)):
            raise WebPluginError(
                f"{where}: path must be the URL of an .html file, such as /name.html or "
                "/route/name.html, and not under /api/ or /fonts/"
            )
        if (not isinstance(self.label, str) or not self.label.strip() or len(self.label) > 80
                or any(character in self.label for character in "{}\r\n")):
            raise WebPluginError(f"{where}: label must be 1 to 80 characters of English without braces or line breaks")
        if not isinstance(self.client, str) or not _CLIENT_RE.fullmatch(self.client):
            raise WebPluginError(f"{where}: client must be 1 to 64 characters of a-z, 0-9, _ and -")
        if not isinstance(self.themed, bool):
            raise WebPluginError(f"{where}: themed must be true or false")
        self._check_csp(where)
        self._check_i18n(where)

    def _check_csp(self, where):
        if not isinstance(self.csp, tuple):
            raise WebPluginError(f"{where}: csp must be a mapping of directive to sources")
        seen = set()
        for relaxation in self.csp:
            if not (isinstance(relaxation, tuple) and len(relaxation) == 2 and isinstance(relaxation[0], str)
                    and isinstance(relaxation[1], tuple) and relaxation[1]
                    and all(isinstance(source, str) for source in relaxation[1])):
                raise WebPluginError(f"{where}: csp must map each directive to a list of sources")
            directive, sources = relaxation
            allowed = ALLOWED_CSP_RELAXATIONS.get(directive)
            if allowed is None or directive in seen:
                raise WebPluginError(
                    f"{where}: csp may not relax '{directive}'; the Viewer allows "
                    + ", ".join(f"{name} {' '.join(values)}" for name, values in ALLOWED_CSP_RELAXATIONS.items())
                )
            seen.add(directive)
            refused = [source for source in sources if source not in allowed]
            if refused or len(set(sources)) != len(sources):
                raise WebPluginError(
                    f"{where}: csp may not add {', '.join(refused) or 'a source twice'} to '{directive}'; "
                    f"the Viewer allows only {' '.join(allowed)}"
                )

    def _check_i18n(self, where):
        if self.i18n_context is None:
            if self.i18n_files:
                raise WebPluginError(f"{where}: i18n files need an i18n context")
            return
        if not isinstance(self.i18n_context, str) or not _CONTEXT_RE.fullmatch(self.i18n_context) \
                or self.i18n_context == MESSAGE_CONTEXT:
            raise WebPluginError(
                f"{where}: i18n context must be a name of letters, digits and underscores, "
                f"and not '{MESSAGE_CONTEXT}', which the console messages use"
            )
        if (not isinstance(self.i18n_files, tuple) or not self.i18n_files
                or not all(isinstance(name, str) and _I18N_FILE_RE.fullmatch(name) for name in self.i18n_files)
                or len(set(self.i18n_files)) != len(self.i18n_files)):
            raise WebPluginError(
                f"{where}: i18n files must be a list of paths relative to src (without .. or a drive), "
                "each file once, in which * and ? name several files"
            )

    @property
    def title(self) -> Message:
        """The Message that names the page."""
        return PageLabel(self.label)

    @property
    def csp_additions(self) -> dict:
        """The relaxations as {directive: sources joined by spaces}, in the manifest's order."""
        return {directive: " ".join(sources) for directive, sources in self.csp}


@dataclass(frozen=True)
class WebPluginDescriptor:
    api_version: int
    plugin_id: str
    backend: str
    register_callable: str
    source_path: str
    pages: tuple = ()

    @property
    def web_root(self) -> str:
        """The folder the web layer's pages are in: the one that holds this descriptor's plugins folder."""
        return os.path.dirname(os.path.dirname(self.source_path))

    @property
    def src_root(self) -> str:
        """The folder of the program's source, which a page's i18n files are relative to."""
        return os.path.dirname(self.web_root)


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


def _check_keys(entry: dict, required, optional, what: str):
    if not all(isinstance(key, str) for key in entry):
        raise WebPluginError(f"{what} field names must be strings")
    keys = frozenset(entry)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise WebPluginError(f"missing {what} field(s): {', '.join(missing)}")
    if unknown:
        raise WebPluginError(f"unknown {what} field(s): {', '.join(unknown)}")


def _page_from_entry(plugin_id: str, entry) -> WebPageDescriptor:
    if not isinstance(entry, dict):
        raise WebPluginError("each page must be a dictionary")
    _check_keys(entry, _PAGE_REQUIRED_KEYS, _PAGE_OPTIONAL_KEYS, "page")
    csp = entry.get("csp", {})
    if not isinstance(csp, dict) or not all(isinstance(key, str) for key in csp):
        raise WebPluginError("csp must be a dictionary of directive to a list of sources")
    relaxations = []
    for directive, sources in csp.items():
        if not isinstance(sources, (list, tuple)):
            raise WebPluginError("csp must be a dictionary of directive to a list of sources")
        relaxations.append((directive, tuple(sources)))
    context, files = None, ()
    if "i18n" in entry:
        i18n = entry["i18n"]
        if not isinstance(i18n, dict):
            raise WebPluginError("i18n must be a dictionary of context and files")
        _check_keys(i18n, _I18N_KEYS, frozenset(), "i18n")
        context = i18n["context"]
        if not isinstance(i18n["files"], (list, tuple)):
            raise WebPluginError("i18n files must be a list of paths")
        files = tuple(i18n["files"])
    return WebPageDescriptor(
        plugin_id=plugin_id,
        path=entry["path"],
        label=entry["label"],
        client=entry["client"],
        themed=entry.get("themed", False),
        csp=tuple(relaxations),
        i18n_context=context,
        i18n_files=files,
    )


def read_plugin_descriptor(path: str) -> WebPluginDescriptor:
    """Read and validate one descriptor without importing its backend."""
    manifest = _literal_manifest(path)
    _check_keys(manifest, _REQUIRED_MANIFEST_KEYS, _OPTIONAL_MANIFEST_KEYS, "manifest")

    api_version = manifest["api_version"]
    plugin_id = manifest["id"]
    backend = manifest["backend"]
    register_callable = manifest["register"]

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
    if not isinstance(register_callable, str) or not _CALLABLE_RE.fullmatch(register_callable):
        raise WebPluginError("register must be a Python callable name")

    entries = manifest.get("pages", [])
    if not isinstance(entries, (list, tuple)):
        raise WebPluginError("pages must be a list")
    pages = tuple(_page_from_entry(plugin_id, entry) for entry in entries)
    for kind, values in (
        ("path", [page.path for page in pages]),
        ("client", [page.client for page in pages]),
    ):
        if len(set(values)) != len(values):
            raise WebPluginError(f"two pages of the plugin share a {kind}")

    return WebPluginDescriptor(
        api_version=api_version,
        plugin_id=plugin_id,
        backend=backend,
        register_callable=register_callable,
        source_path=os.path.abspath(path),
        pages=pages,
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


def bundled_plugin_dir() -> str:
    """The folder of the bundled plugins' manifests."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")


def page_file_key(path) -> str:
    """The key the registry's page tables use for the file at path: absolute and case-normalized."""
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def i18n_files(page: WebPageDescriptor, src_root: str) -> list:
    """The files, relative to src_root and with /, that hold page's texts (its patterns expanded, in order).

    A pattern that names no file names none: the check that a page's files
    are there is a test's.
    """
    found = []
    root = os.path.abspath(src_root)
    for pattern in page.i18n_files:
        # The folder is a path, not a pattern: a [ or * in its name is its own.
        for match in sorted(glob.glob(os.path.join(glob.escape(root), *pattern.split("/")))):
            if not os.path.isfile(match):
                continue
            try:
                inside = os.path.commonpath([root, os.path.abspath(match)]) == root
            except ValueError:
                inside = False
            relative = os.path.relpath(match, root).replace(os.sep, "/")
            if inside and relative not in found:
                found.append(relative)
    return found


def contexts_of(descriptors, src_root: str) -> dict:
    """{file relative to src_root: catalog context} for the pages of descriptors.

    Raises WebPluginError when two pages claim a file.
    """
    contexts, owners = {}, {}
    for descriptor in descriptors:
        for page in descriptor.pages:
            for relative in i18n_files(page, src_root):
                if relative in owners and owners[relative] != page.path:
                    raise WebPluginError(
                        f"'{relative}' holds the texts of pages '{owners[relative]}' and '{page.path}'"
                    )
                owners[relative] = page.path
                contexts[relative] = page.i18n_context
    return contexts


def page_contexts(src_root: str):
    """{file relative to src_root: catalog context} for every page the plugins under src_root declare.

    Read from the manifests in <src_root>/web_ui/plugins without importing
    anything, as the update of the translations needs it. Returns it with the
    diagnostics of the manifests that could not be read, and raises
    WebPluginError when two pages claim a file.
    """
    plugin_dir = os.path.join(src_root, "web_ui", "plugins")
    if not os.path.isdir(plugin_dir):
        return {}, []
    descriptors, diagnostics = discover_plugin_descriptors(plugin_dir)
    return contexts_of(descriptors, src_root), diagnostics


@dataclass(frozen=True)
class ApiRequest:
    """What a plugin's API handler is given: the request after the web server's own checks.

    path is the request's path without its query; headers are the request's;
    body is what a POST sent, at most the route's max_body bytes (b"" for a GET).
    """

    method: str
    path: str
    query: str
    headers: object
    body: bytes = b""


@dataclass(frozen=True)
class ApiRoute:
    plugin_id: str
    path: str
    method: str
    handler: Callable
    max_body: int | None = None
    content_types: tuple | None = None
    size_error: object = None

    @property
    def is_prefix(self) -> bool:
        return self.path.endswith("/")


def match_api_route(routes: Mapping, method: str, path: str):
    """The route in routes that answers method and path: an exact path first, then the longest prefix."""
    route = routes.get((method, path))
    if route is not None:
        return route
    best = None
    for (route_method, route_path), candidate in routes.items():
        if (route_method == method and candidate.is_prefix and path.startswith(route_path)
                and (best is None or len(route_path) > len(best.path))):
            best = candidate
    return best


class WebPluginRegistry:
    """Per-Viewer routes, actions, state providers, pages, and plugin ownership."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.actions: dict[str, Callable] = {}
        self.static_routes: dict[str, str] = {}
        self.api_routes: dict[tuple[str, str], ApiRoute] = {}
        self.pages: dict[str, WebPageDescriptor] = {}
        self._action_owners: dict[str, str] = {}
        self._route_owners: dict[str, str] = {}
        self._state_providers: dict[tuple[str, str], Callable] = {}
        self._page_files: dict[str, WebPageDescriptor] = {}
        self._i18n_files: dict[str, tuple[str, str]] = {}
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

    def register_api_route(
        self,
        plugin_id: str,
        path: str,
        handler: Callable,
        methods=("GET",),
        *,
        max_body: int | None = None,
        content_types=None,
        size_error=None,
    ):
        """Answer requests for path under /api/<plugin_id>/ with handler(ApiRequest).

        A path that ends with / is a prefix, answered for every path below it;
        any other path answers only itself. The handler returns (status,
        payload) or (status, payload, headers), and the server sends the payload
        as JSON. It runs after the server's own checks, which no route skips:
        the same-origin guard, and for a POST the size of the body (max_body,
        required, and the body must have 1 to max_body bytes) and, when
        content_types is given, its Content-Type. A POST whose body is too
        large or missing is answered 413 with size_error (a Message or text),
        and a ValueError the handler raises is answered 400 with its text.
        ``/api/`` is otherwise the server's: only a plugin's own folder of it can be
        registered, and a plugin may not take the name of a core route.
        """
        self._validate_registration_owner(plugin_id)
        if plugin_id in CORE_API_NAMES:
            raise WebPluginError(f"plugin '{plugin_id}' cannot register API routes: /api/{plugin_id}/ is the server's")
        base = f"/api/{plugin_id}/"
        if (not isinstance(path, str) or not path.startswith(base) or not _API_PATH_CHARACTERS.fullmatch(path)
                or "//" in path or any(segment in (".", "..") for segment in path.split("/"))):
            raise WebPluginError(f"API route '{path}' must be under {base}, made of plain path segments")
        if not callable(handler):
            raise WebPluginError(f"API route '{path}' handler is not callable")
        if isinstance(methods, str):
            methods = (methods,)
        methods = tuple(methods)
        if not methods or not all(isinstance(method, str) and method in API_METHODS for method in methods) \
                or len(set(methods)) != len(methods):
            raise WebPluginError(f"API route '{path}' methods must be GET and/or POST, each once")
        if "POST" in methods:
            if isinstance(max_body, bool) or not isinstance(max_body, int) or max_body <= 0:
                raise WebPluginError(f"API route '{path}' takes POST, so it needs max_body, the largest body in bytes")
        elif max_body is not None or content_types is not None:
            raise WebPluginError(f"API route '{path}' takes no body, so max_body and content_types do not apply")
        if content_types is not None:
            if isinstance(content_types, str):
                content_types = (content_types,)
            content_types = tuple(content_types)
            if not content_types or not all(
                isinstance(name, str) and name == name.strip().lower() and "/" in name for name in content_types
            ):
                raise WebPluginError(f"API route '{path}' content_types must be lowercase media types")
        for method in methods:
            existing = self.api_routes.get((method, path))
            if existing is not None and existing.plugin_id != plugin_id:
                raise WebPluginError(
                    f"API route {method} '{path}' is already registered by plugin '{existing.plugin_id}'"
                )
        # A plugin registering its own route again replaces it, as for an action:
        # register_backend runs again whenever the plugin's command does.
        for method in methods:
            self.api_routes[(method, path)] = ApiRoute(
                plugin_id, path, method, handler,
                max_body=max_body if method == "POST" else None,
                content_types=content_types if method == "POST" else None,
                size_error=size_error,
            )

    def register_state_provider(
        self, plugin_id: str, name: str, provider: Callable
    ):
        self._validate_registration_owner(plugin_id)
        if not isinstance(name, str) or not name.strip():
            raise WebPluginError("state-provider name must be a non-empty string")
        if not callable(provider):
            raise WebPluginError(f"state provider '{name}' is not callable")
        self._state_providers[(plugin_id, name)] = provider

    def _page_file(self, page: WebPageDescriptor, web_root: str) -> str:
        """The file the web server serves page.path from, and the checks that it is the page's plugin's own."""
        for prefix, folder in self.static_routes.items():
            if page.path.startswith(prefix):
                owner = self._route_owners.get(prefix)
                if owner != page.plugin_id:
                    raise WebPluginError(
                        f"page '{page.path}' is in the route '{prefix}' of plugin '{owner}', not of its own"
                    )
                return page_file_key(os.path.join(folder, *page.path[len(prefix):].split("/")))
        first = page.path[1:].split("/")[0]
        stem = first[:-len(".html")] if first.endswith(".html") else first
        if stem != page.plugin_id and not stem.startswith(page.plugin_id + "_"):
            raise WebPluginError(
                f"page '{page.path}' is not in a route of plugin '{page.plugin_id}'; a page of the web "
                f"folder is named for its plugin: /{page.plugin_id}.html or /{page.plugin_id}_<name>.html"
            )
        return page_file_key(os.path.join(web_root, *page.path[1:].split("/")))

    def register_page(self, page: WebPageDescriptor, *, web_root: str, src_root: str):
        """Take page, a plugin's page, into the tables the server reads.

        Its file is where the web server serves its path from: a static route
        of its plugin (register that first), or web_root, the web layer's own
        folder. The same file or client may not be declared twice, a context
        of the catalog belongs to one plugin, and a file of texts to one page.
        """
        self._validate_registration_owner(page.plugin_id)
        if page.client in self.pages:
            raise WebPluginError(f"page '{page.path}': client '{page.client}' is already a page's")
        if any(existing.path == page.path for existing in self.pages.values()):
            raise WebPluginError(f"page '{page.path}' is already registered")
        file_key = self._page_file(page, web_root)
        if file_key in self._page_files:
            raise WebPluginError(
                f"page '{page.path}': its file is already the page '{self._page_files[file_key].path}'s"
            )
        texts = i18n_files(page, src_root) if page.i18n_context else []
        keys = [page_file_key(os.path.join(src_root, *relative.split("/"))) for relative in texts]
        for key in keys:
            owner = self._i18n_files.get(key)
            if owner is not None:
                raise WebPluginError(f"page '{page.path}': '{key}' already holds the texts of page '{owner[1]}'")
        if page.i18n_context:
            for other in self.pages.values():
                if other.i18n_context == page.i18n_context and other.plugin_id != page.plugin_id:
                    raise WebPluginError(
                        f"page '{page.path}': the catalog context '{page.i18n_context}' belongs to plugin '{other.plugin_id}'"
                    )
        self.pages[page.client] = page
        self._page_files[file_key] = page
        for key in keys:
            self._i18n_files[key] = (page.i18n_context, page.path)

    def page(self, client: str) -> WebPageDescriptor:
        """The page whose event client is ``client``; WebPageUnavailable when no plugin registered one."""
        try:
            return self.pages[client]
        except KeyError:
            raise WebPageUnavailable(f"No plugin registered a web page for the client '{client}'.") from None

    def csp_additions(self, filepath) -> dict:
        """The relaxations of the Content-Security-Policy that the page in the file filepath declared."""
        page = self._page_files.get(page_file_key(filepath))
        return page.csp_additions if page is not None else {}

    def is_themed(self, filepath) -> bool:
        """Whether the page in the file filepath declared that it takes the Studio theme."""
        page = self._page_files.get(page_file_key(filepath))
        return page is not None and page.themed

    def text_context(self, filepath):
        """The catalog context of the texts in the file filepath, or None for a file no page claimed."""
        entry = self._i18n_files.get(page_file_key(filepath))
        return entry[0] if entry is not None else None

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
            dict(self.api_routes),
            dict(self.pages),
            dict(self._page_files),
            dict(self._i18n_files),
        )

    def restore(self, snapshot):
        (actions, routes, action_owners, route_owners, providers, registered,
         api_routes, pages, page_files, i18n_files_) = snapshot
        self.actions.clear()
        self.actions.update(actions)
        self.static_routes.clear()
        self.static_routes.update(routes)
        self.api_routes.clear()
        self.api_routes.update(api_routes)
        self.pages.clear()
        self.pages.update(pages)
        self._action_owners = action_owners
        self._route_owners = route_owners
        self._state_providers = providers
        self._page_files = page_files
        self._i18n_files = i18n_files_
        self.registered_plugins = registered


_bundled_pages_registry = None


def register_bundled_pages(registry: WebPluginRegistry) -> WebPluginRegistry:
    """Take the pages the bundled manifests declare into registry, without their backends; returns registry.

    A page that lives in a route of its plugin needs that route, which only
    the plugin's backend registers, so such a page is left out here.
    """
    descriptors, _ = discover_plugin_descriptors(bundled_plugin_dir())
    for descriptor in descriptors:
        for page in descriptor.pages:
            try:
                registry.register_page(page, web_root=descriptor.web_root, src_root=descriptor.src_root)
            except WebPluginError:
                continue
    return registry


def bundled_pages() -> WebPluginRegistry:
    """A registry holding only the pages the bundled manifests declare.

    What the web server falls back on when its Viewer has no registry of its
    own, as in tests (register_bundled_pages fills a registry of a test's own).
    """
    global _bundled_pages_registry
    if _bundled_pages_registry is None:
        _bundled_pages_registry = register_bundled_pages(WebPluginRegistry(None))
    return _bundled_pages_registry


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
        self.plugin_dir = plugin_dir or bundled_plugin_dir()
        self.descriptors: dict[str, WebPluginDescriptor] = {}
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
            if not callable(register_callable):
                raise WebPluginError("declared registration callable is not callable")
        except Exception as error:
            self._record_error(descriptor, "backend import", error)
            return False

        snapshot = self.registry.snapshot()
        try:
            self.registry._active_registration_owner = descriptor.plugin_id
            register_callable(self.registry, self.viewer)
            for page in descriptor.pages:
                self.registry.register_page(
                    page, web_root=descriptor.web_root, src_root=descriptor.src_root
                )
            self.registry.registered_plugins.add(descriptor.plugin_id)
            return True
        except Exception as error:
            self.registry.restore(snapshot)
            self._record_error(descriptor, "backend registration", error)
            return False
        finally:
            self.registry._active_registration_owner = None
