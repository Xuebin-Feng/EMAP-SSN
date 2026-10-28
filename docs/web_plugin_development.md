# Bundled Web-Utility Plugins

The Viewer discovers trusted web-utility descriptors from
`src/web_ui/plugins/` once during startup. Discovery parses each descriptor with
Python's AST and does not import its backend until the manifest has passed
validation. User-installed plugin directories and runtime hot reloading are not
supported.

A plugin declares itself. Its manifest says which pages it serves and what each
page needs; its backend registers what the page talks to. The web server, the
Viewer's page opener and the translation update read those declarations, so
adding a plugin changes none of them.

## Descriptor contract

Each descriptor is a Python file containing one literal `WEB_PLUGIN`
dictionary and no required executable code:

```python
WEB_PLUGIN = {
    "api_version": 1,
    "id": "example",
    "backend": "web_ui.example_backend",
    "register": "register_backend",
    "pages": [
        {
            "path": "/example.html",
            "label": "Example UI",
            "client": "example",
            "themed": True,
            "csp": {"connect-src": ["https:"]},
            "i18n": {
                "context": "ExamplePage",
                "files": ["web_ui/example.html", "web_ui/example_*.js"],
            },
        },
    ],
}
```

The four fields `api_version`, `id`, `backend` and `register` are required, and
`pages` is optional (a plugin may serve actions and API routes only). Unknown
fields are rejected. The ID must begin with a lowercase letter and contain only
lowercase letters, digits, and underscores. Plugin IDs must be unique.

The manifest stays a literal. It is read without importing anything, so tools
that run without the application (the translation update, the tests) read it
too.

### Pages

Each entry of `pages` declares one browser page. Unknown fields are rejected, and
every value is checked.

| Field | Required | Meaning |
| --- | --- | --- |
| `path` | yes | The URL the page is served at: an `.html` file made of plain path segments, not under `/api/` or `/fonts/`. It is a file of the web folder `src/web_ui/` named for the plugin (`/<id>.html` or `/<id>_<name>.html`), or a file of one of the plugin's own static routes (`/example_resource/page.html`); it cannot be in another plugin's route. |
| `label` | yes | What the opener's console lines and dialogs call the page, in English: up to 80 characters, without braces or line breaks. It is shown translated, as a console `Message`, and the translation update lists it from the manifest. |
| `client` | yes | The id the page's event stream connects with (`/api/events?client=<id>`): 1 to 64 characters of `a-z`, `0-9`, `_` and `-`. It is how the Viewer tells that the page is open, and how `viewer.open_page` names it. One page has one client. |
| `themed` | no | `True` for a page that takes the Viewer's Studio theme colour: the server ends its head with the theme's custom properties. Default `False`. |
| `csp` | no | The relaxations of the base Content-Security-Policy the page needs, as `{directive: [sources]}`. Only the relaxations the Viewer allows are accepted (see below). |
| `i18n` | no | `{"context": ..., "files": [...]}`: the catalog context of the page's texts, and the files that hold them, relative to `src/`. A name may hold `*` and `?` to name several files, as in `web_ui/example_*.js`. A file belongs to one page, and a context to one plugin. |

A plugin whose page marks its texts (`data-i18n`, `t("...")`; see
`web_ui/Page_Texts.py`) must name every file that holds them, or the page stays
in English. The translation update fails when a manifest cannot be read, and
`tests/test_web_plugin_discovery.py` fails when a script with marked texts is
declared by no page.

The tables the server keeps by file (the policy, the theme and the texts) are
keyed by the page's absolute path, so a page's relaxations belong to the file the
plugin serves it from: a file of the same name that another route or folder
serves does not get them.

### The backend

The backend must expose the declared `register` callable:

```python
from utilities.Localization import Message
from web_ui.Plugin_Manager import ensure_registry


def register_backend(registry, viewer):
    registry.register_action(
        "example", "example_action", lambda data: handle_action(viewer, data)
    )
    registry.register_static_route(
        "example", "/example_resource/", EXAMPLE_RESOURCE_DIRECTORY
    )
    registry.register_api_route(
        "example", "/api/example/check", handle_check,
        methods=("POST",), max_body=1024 * 1024,
        size_error=Message("The file must be nonempty and at most 1 MiB."),
    )
    registry.register_state_provider("example", "state", extend_initial_state)


def handle_check(request):
    # request.method, .path, .query, .headers and .body (the bytes of a POST)
    return 200, {"size": len(request.body)}


def activate(viewer):
    viewer.add_console_button(
        "exampleBtn", "Example", lambda: viewer.open_page("example"), "Open Example", icon="sparkles"
    )


def register(viewer):
    # Compatibility entry point for commands and existing imports.
    registry = ensure_registry(viewer)
    register_backend(registry, viewer)
    registry.registered_plugins.add("example")
    return activate(viewer)
```

`register_backend` is called automatically and must be idempotent. It should
only connect backend capabilities; it must not open a browser, add a Console
button, create output directories, start a model, or perform other activation
work.

`activate` is not part of the manifest. The Viewer never calls it when it starts:
the plugin's command calls it, through the backend's `register(viewer)`, when the
command runs (a `<command> --register-only` also does, when a session loads). It
retains command-driven UI behavior and may be called more than once, so the
Viewer-side button identifier must remain stable. An earlier manifest named it in
an `activate` field; the field did nothing and is now refused.

To open a page, call `viewer.open_page("<client>")`, or
`viewer.open_page("<client>", show_existing_dialog=False)` where nobody is at the
Viewer to dismiss the dialog that says the page is already open. The address, the
label and the client are the manifest's. The call raises
`WebPageUnavailable` when no plugin registered a page of that client, and returns
whether the page opened.

`add_console_button(name, label, callback, tooltip=None, icon=None)` registers a
button for the second header row of the Console window, which shows each
registered button with its icon (the name of a bundled Lucide icon) and its
label, in the order they were registered. `label` and `tooltip` are translated
texts, and a click calls `callback` with no argument. A button registered while
the Console is open appears in it at once, and one registered before it opens is
there when it does; the row is not shown while nothing is registered. A call that
finds the name already registered changes nothing. The Console window names no
page or command: it shows what is registered. A plugin that should come back when
a session loads adds the name of its command to `viewer.sidebar_buttons_to_persist`
(the list keeps its name from the days the buttons were in a sidebar), and the
Viewer runs `<command> --register-only` for each name. The Viewer's old sidebar,
with `add_sidebar_button`, is kept in the code for a later use but never opens,
so a plugin does not use it.

## Registration rules

- Action names are global. A second plugin cannot replace an action owned by a
  different plugin.
- Static route prefixes must begin with `/` and are normalized to end with `/`.
  `/api/` and `/fonts/` are reserved, and route prefixes from different plugins
  cannot overlap.
- A static route, like the bundled pages' own folder `src/web_ui/`, sends only
  the kinds of file in `MIME_TYPES` (`Web_Server.py`): pages, scripts,
  stylesheets, JSON, images, fonts, Markdown and `.pdb` structures. A file of
  any other extension in its directory, a Python source among them, is
  answered as a missing file. A plugin whose page loads another kind adds its
  extension and content type to `MIME_TYPES`.
- An API route is registered with
  `register_api_route(plugin_id, path, handler, methods=("GET",), *, max_body=None, content_types=None, size_error=None)`.
  Its path is under `/api/<plugin_id>/`, which is the only part of `/api/` a
  plugin may take: `/api/action`, `/api/events` and `/api/mcp/` are the
  server's, and a plugin of one of those names registers no API route. A path that
  ends with `/` answers everything below it, and any other path only itself; an
  exact path wins, then the longest prefix. `methods` are `GET` and/or `POST`. The
  handler gets an `ApiRequest` and answers `(status, payload)` or
  `(status, payload, headers)`, and the server sends the payload as JSON. A
  `ValueError` it raises is a 400 with its text in the Viewer's language, and any
  other error a 500. A plugin that registers its own route again replaces it, as
  `register_backend` runs again when the plugin's command does.
- A state provider receives `(viewer, state)` and must return a mapping. State
  providers run in deterministic `(plugin_id, provider_name)` order whenever an
  SSE client requests initial Viewer state. The Viewer's own state holds the
  network and the metadata; what only a plugin's page needs (the Agent's model, for
  one) is that plugin's provider's.
- Registration is atomic per plugin. If import or registration fails, all
  routes, actions, pages and state providers added by that plugin are rolled back;
  startup continues with the remaining plugins and prints a diagnostic.
- Static routes may point to an output directory that does not yet exist. Create
  output directories only when the corresponding operation runs.

## What stays central, and why

The plugins declare what is theirs. What protects the Viewer stays in the core,
because a plugin cannot be trusted to hold itself to it:

- **The same-origin guard.** Every route, plugin or not, is refused with a 403
  when its `Origin`, `Sec-Fetch-Site` or `Host` does not name the Viewer, before
  anything is routed (`Web_Server.py`). A web page the user visits could
  otherwise drive an action through a plain cross-origin request, or a rebound DNS
  name read the Viewer's state.
- **The size and the type of a body.** For a POST, the server checks the body's
  length (`max_body`, and a body must have at least a byte) and, if the route
  names them, its `Content-Type`, before the handler runs; a body it refuses is not
  read into memory.
- **The Content-Security-Policy.** The base policy is the server's
  (`Web_Server.PAGE_POLICY`), and a page may relax it only as far as
  `Plugin_Manager.ALLOWED_CSP_RELAXATIONS`: `'unsafe-eval'` in `script-src`
  (Mol*'s encoder), and `https:` and `data:` in `connect-src`. A manifest that asks
  for anything else is refused with a message naming what it asked for. Inline
  scripts are always allowed by their hash only.
- **Where files are served from.** Static routes and the web folder serve a file
  only through `Web_Server.contained_path` (no `..`, no links out), and only kinds
  of file in `MIME_TYPES`.
- **`/api/action`, the event stream and the MCP bridge.** The page-to-Viewer
  action channel dispatches to the registered actions, and its content-type rule
  and size limit are the server's; `/api/mcp/` is authenticated by the Viewer's
  token. A plugin registers actions and routes, not these.
- **The reserved prefixes** `/api/` and `/fonts/`, and which plugin may take which
  name (a page of the web folder is named for its plugin; a page in a route is in
  its own plugin's).

## Adding a bundled utility

1. Implement the backend's `register_backend` (and `activate`, `register` if a
   command or a button brings the utility up).
2. Add its literal descriptor under `src/web_ui/plugins/`, with its pages, and put
   the page's files where the descriptor says.
3. Add or update the Viewer command that activates the utility when requested.
4. Add focused tests for discovery, registration, initial state, command
   activation, and any routes or actions introduced by the plugin.

No edit to `EMAPSSN_Viewer.py`, `Web_Server.py`, `Page_Texts.py` or
`Update_Translations.py` is required for subsequent bundled utilities that follow
this contract: the server's policy, theme and texts, the translation catalog's
page texts and labels, and the Viewer's page opener are read from the manifest.
`tests/test_web_plugin_discovery.py` serves a plugin that exists only in a
temporary tree through the real server to show it, and scans the central web code
for the name of any bundled page, plugin or backend.
