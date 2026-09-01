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
    viewer.add_sidebar_button(
        "exampleBtn", "Example", viewer.open_example_ui, "Open Example"
    )


def register(viewer):
    # Compatibility entry point for commands and existing imports.
    registry = ensure_registry(viewer)
    register_backend(registry, viewer)
    registry.registered_plugins.add("example")
    return activate(viewer)
```

`register_backend` is called automatically and must be idempotent. It should
only connect backend capabilities; it must not open a browser, show a sidebar
button, create output directories, start a model, or perform other activation
work. `activate` retains command-driven UI behavior and may be called more than
once, so the Viewer-side button identifier must remain stable.

## Registration rules

- Action names are global. A second plugin cannot replace an action owned by a
  different plugin.
- Static route prefixes must begin with `/` and are normalized to end with `/`.
  `/api/` and `/fonts/` are reserved, and route prefixes from different plugins
  cannot overlap.
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
