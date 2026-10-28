# Bundled Web-Utility Plugins

The Viewer discovers trusted web-utility descriptors from
`src/web_ui/plugins/` once during startup. Discovery parses each descriptor with
Python's AST and does not import its backend until the manifest has passed
validation. User-installed plugin directories and runtime hot reloading are not
supported.

## Descriptor contract

Each descriptor is a Python file containing one literal `WEB_PLUGIN`
dictionary and no required executable code:

```python
WEB_PLUGIN = {
    "api_version": 1,
    "id": "example",
    "backend": "web_ui.example_backend",
    "register": "register_backend",
    "activate": "activate",
}
```

The five fields shown above are required; unknown fields are rejected. The ID
must begin with a lowercase letter and contain only lowercase letters, digits,
and underscores. Plugin IDs must be unique.

The backend must expose the declared callables:

```python
from web_ui.Plugin_Manager import ensure_registry


def register_backend(registry, viewer):
    registry.register_action(
        "example", "example_action", lambda data: handle_action(viewer, data)
    )
    registry.register_static_route(
        "example", "/example_resource/", EXAMPLE_RESOURCE_DIRECTORY
    )
    registry.register_state_provider("example", "state", extend_initial_state)


def activate(viewer):
    viewer.add_console_button(
        "exampleBtn", "Example", viewer.open_example_ui, "Open Example", icon="sparkles"
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
work. `activate` retains command-driven UI behavior and may be called more than
once, so the Viewer-side button identifier must remain stable.

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
- A state provider receives `(viewer, state)` and must return a mapping. State
  providers run in deterministic `(plugin_id, provider_name)` order whenever an
  SSE client requests initial Viewer state.
- Registration is atomic per plugin. If import or registration fails, all
  routes, actions, and state providers added by that plugin are rolled back;
  startup continues with the remaining plugins and prints a diagnostic.
- Static routes may point to an output directory that does not yet exist. Create
  output directories only when the corresponding operation runs.

## Adding a bundled utility

1. Implement the backend registration and activation callables.
2. Add its literal descriptor under `src/web_ui/plugins/`.
3. Add or update the Viewer command that activates the utility when requested.
4. Add focused tests for discovery, registration, initial state, command
   activation, and any routes or actions introduced by the plugin.

No edit to `EMAPSSN_Viewer.py` or `Web_Server.py` is required for subsequent
bundled utilities that follow this contract.
