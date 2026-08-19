"""Regression tests for bundled web-plugin discovery and registration."""

from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from web_ui import Plugin_Manager  # noqa: E402
from web_ui.Plugin_Manager import (  # noqa: E402
    WebPluginError,
    WebPluginManager,
    WebPluginRegistry,
    discover_plugin_descriptors,
)


class FakeViewer:
    def __init__(self):
        self.web_action_handlers = {}
        self.metadata = {
            "Length": {"type": "number", "values": []},
            "Organism": {"type": "text", "values": []},
        }
        self.buttons = []
        self.sidebar_buttons_to_persist = []

    def add_sidebar_button(self, *args, **kwargs):
        self.buttons.append((args, kwargs))

    def open_agent_ui(self):
        return None

    def open_metadata_ui(self):
        return None

    def get_initial_web_state(self):
        return {"base": True}


class ManifestDiscoveryTests(unittest.TestCase):
    @staticmethod
    def _write_descriptor(folder, name, **overrides):
        manifest = {
            "api_version": 1,
            "id": name,
            "backend": f"fake_backends.{name}",
            "register": "register_backend",
            "activate": "activate",
        }
        manifest.update(overrides)
        path = Path(folder) / f"{name}.py"
        path.write_text(f"WEB_PLUGIN = {manifest!r}\n", encoding="utf-8")
        return path

    def test_bundled_descriptors_are_discovered_without_importing_backends(self):
        plugin_dir = SRC_DIR / "web_ui" / "plugins"
        with mock.patch.object(Plugin_Manager.importlib, "import_module") as importer:
            descriptors, diagnostics = discover_plugin_descriptors(str(plugin_dir))

        self.assertEqual(
            [descriptor.plugin_id for descriptor in descriptors],
            ["agent", "esmfold", "meta"],
        )
        self.assertEqual(diagnostics, [])
        importer.assert_not_called()

    def test_manifest_errors_and_duplicate_ids_do_not_hide_valid_plugins(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_descriptor(temp_dir, "alpha")
            self._write_descriptor(temp_dir, "duplicate_a", id="alpha")
            self._write_descriptor(temp_dir, "future", api_version=99)
            (Path(temp_dir) / "malformed.py").write_text(
                "WEB_PLUGIN = dict(id='malformed')\n", encoding="utf-8"
            )
            self._write_descriptor(temp_dir, "omega")

            descriptors, diagnostics = discover_plugin_descriptors(temp_dir)

        self.assertEqual(
            [descriptor.plugin_id for descriptor in descriptors], ["alpha", "omega"]
        )
        messages = [diagnostic.message for diagnostic in diagnostics]
        self.assertTrue(any("duplicate plugin id" in message for message in messages))
        self.assertTrue(any("unsupported API version" in message for message in messages))
        self.assertTrue(any("literal dictionary" in message for message in messages))


class RegistryTests(unittest.TestCase):
    def test_action_and_route_ownership_prevents_silent_overwrites(self):
        registry = WebPluginRegistry(object())
        registry.register_action("alpha", "shared", lambda data: data)
        registry.register_static_route("alpha", "/alpha/", ".")

        with self.assertRaises(WebPluginError):
            registry.register_action("beta", "shared", lambda data: data)
        with self.assertRaises(WebPluginError):
            registry.register_static_route("beta", "/alpha/nested/", ".")
        with self.assertRaises(WebPluginError):
            registry.register_static_route("beta", "/api/plugin/", ".")
        with self.assertRaises(WebPluginError):
            registry.register_static_route("beta", "/fonts/custom/", ".")

    def test_state_providers_are_ordered_and_idempotently_replaceable(self):
        registry = WebPluginRegistry(object())
        registry.register_state_provider(
            "beta", "state", lambda viewer, state: {**state, "order": state["order"] + "b"}
        )
        registry.register_state_provider(
            "alpha", "state", lambda viewer, state: {**state, "order": state["order"] + "a"}
        )
        registry.register_state_provider(
            "alpha", "state", lambda viewer, state: {**state, "order": state["order"] + "A"}
        )

        self.assertEqual(registry.apply_state_providers({"order": ""})["order"], "Ab")
        self.assertEqual(len(registry._state_providers), 2)

    def test_server_uses_the_preconfigured_registry_route_mapping(self):
        from web_ui import Web_Server

        viewer = FakeViewer()
        registry = WebPluginRegistry(viewer)
        viewer.web_plugin_registry = registry
        registry.register_static_route("alpha", "/alpha/", ".")
        server = Web_Server.start_server(viewer, preferred_port=0)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        self.assertIs(server.static_routes, registry.static_routes)
        self.assertIn("/alpha/", server.static_routes)
        self.assertIn("/fonts/", server.static_routes)


class MetadataHighlightActionTests(unittest.TestCase):
    def make_viewer(self):
        return type(
            "MetadataViewer",
            (),
            {
                "n_nodes": 3,
                "visible_mask": np.array([True, True, False], dtype=bool),
                "apply_left_click_focus": mock.Mock(),
                "clear_left_click_focus": mock.Mock(),
            },
        )()

    def test_select_accepts_only_visible_integer_node_indices(self):
        from web_ui import meta_backend

        viewer = self.make_viewer()
        self.assertTrue(meta_backend.handle_select_node(viewer, {"index": 1}))
        viewer.apply_left_click_focus.assert_called_once_with(1)

        for invalid in (True, 1.0, "1", -1, 2, 3, None):
            with self.subTest(index=invalid):
                self.assertFalse(
                    meta_backend.handle_select_node(viewer, {"index": invalid})
                )
        viewer.apply_left_click_focus.assert_called_once_with(1)

    def test_registered_clear_action_only_delegates_click_focus_clear(self):
        from web_ui import meta_backend

        viewer = self.make_viewer()
        registry = WebPluginRegistry(viewer)
        meta_backend.register_backend(registry, viewer)

        self.assertIn("select", registry.actions)
        self.assertIn("clear_selection", registry.actions)
        self.assertTrue(registry.actions["clear_selection"]({}))
        viewer.clear_left_click_focus.assert_called_once_with()

    def test_activation_wrapper_does_not_duplicate_viewer_row_broadcast(self):
        from web_ui import meta_backend

        viewer = SimpleNamespace(
            left_click_highlight_indices=[1],
            broadcast_event=mock.Mock(),
            add_sidebar_button=mock.Mock(),
            open_metadata_ui=mock.Mock(),
            sidebar_buttons_to_persist=[],
        )
        original_mouse_press = mock.Mock(
            side_effect=lambda _event: viewer.broadcast_event(
                {"type": "highlight_row", "index": 1}
            )
        )
        viewer.on_mouse_press = original_mouse_press

        meta_backend.activate(viewer)
        viewer.on_mouse_press(SimpleNamespace(button=1, modifiers=[]))

        original_mouse_press.assert_called_once()
        viewer.broadcast_event.assert_called_once_with(
            {"type": "highlight_row", "index": 1}
        )
        self.assertIsNone(viewer.left_click_highlight_indices)


class ManagerTests(unittest.TestCase):
    @staticmethod
    def _module(register):
        module = ModuleType("fake_backend")
        module.register_backend = register
        module.activate = lambda viewer: None
        return module

    @staticmethod
    def _write_descriptor(folder, filename, plugin_id):
        manifest = {
            "api_version": 1,
            "id": plugin_id,
            "backend": f"fake_backends.{plugin_id}",
            "register": "register_backend",
            "activate": "activate",
        }
        (Path(folder) / filename).write_text(
            f"WEB_PLUGIN = {manifest!r}\n", encoding="utf-8"
        )

    def test_registration_failure_rolls_back_and_does_not_stop_later_plugins(self):
        def register_alpha(registry, viewer):
            registry.register_action("alpha", "shared", lambda data: data)

        def register_bad(registry, viewer):
            registry.register_action("bad", "temporary", lambda data: data)
            raise RuntimeError("intentional failure")

        def register_collision(registry, viewer):
            registry.register_action("collision", "shared", lambda data: data)

        def register_omega(registry, viewer):
            registry.register_action("omega", "omega", lambda data: data)

        modules = {
            "fake_backends.alpha": self._module(register_alpha),
            "fake_backends.bad": self._module(register_bad),
            "fake_backends.collision": self._module(register_collision),
            "fake_backends.omega": self._module(register_omega),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_descriptor(temp_dir, "a_alpha.py", "alpha")
            self._write_descriptor(temp_dir, "b_bad.py", "bad")
            self._write_descriptor(temp_dir, "c_collision.py", "collision")
            self._write_descriptor(temp_dir, "d_omega.py", "omega")
            viewer = FakeViewer()
            manager = WebPluginManager(viewer, plugin_dir=temp_dir)
            with mock.patch.object(
                Plugin_Manager.importlib,
                "import_module",
                side_effect=lambda name: modules[name],
            ):
                manager.discover_and_register()

        self.assertEqual(manager.registry.registered_plugins, {"alpha", "omega"})
        self.assertEqual(set(manager.registry.actions), {"shared", "omega"})
        self.assertNotIn("temporary", manager.registry.actions)
        self.assertEqual(
            {diagnostic.plugin_id for diagnostic in manager.diagnostics},
            {"bad", "collision"},
        )

    def test_backend_cannot_register_capabilities_under_another_plugin_id(self):
        def register_alpha(registry, viewer):
            registry.register_action("impostor", "action", lambda data: data)

        modules = {"fake_backends.alpha": self._module(register_alpha)}
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_descriptor(temp_dir, "alpha.py", "alpha")
            manager = WebPluginManager(FakeViewer(), plugin_dir=temp_dir)
            with mock.patch.object(
                Plugin_Manager.importlib,
                "import_module",
                side_effect=lambda name: modules[name],
            ):
                manager.discover_and_register()

        self.assertEqual(manager.registry.actions, {})
        self.assertEqual(manager.registry.registered_plugins, set())
        self.assertIn("attempted to register", manager.diagnostics[0].message)

    def test_bundled_registration_does_not_activate_sidebar_buttons(self):
        from web_ui import agent_backend, meta_backend

        viewer = FakeViewer()
        original_state_method = viewer.get_initial_web_state.__func__
        manager = WebPluginManager(viewer)
        manager.discover_and_register()

        self.assertEqual(manager.registry.registered_plugins, {"agent", "esmfold", "meta"})
        self.assertEqual(viewer.buttons, [])
        self.assertIs(viewer.get_initial_web_state.__func__, original_state_method)

        with mock.patch.object(agent_backend, "load_agent_history", return_value=[{"role": "user"}]):
            state = manager.registry.apply_state_providers({"base": True})
        self.assertEqual(state["columns"], ["Node ID", "Length", "Organism"])
        self.assertEqual(state["types"], {"Length": "number", "Organism": "text"})
        self.assertEqual(state["llm_history"], [{"role": "user"}])

        action_count = len(manager.registry.actions)
        provider_count = len(manager.registry._state_providers)
        meta_backend.register_backend(manager.registry, viewer)
        agent_backend.register_backend(manager.registry, viewer)
        self.assertEqual(len(manager.registry.actions), action_count)
        self.assertEqual(len(manager.registry._state_providers), provider_count)

        manager.activate("agent")
        self.assertEqual(len(viewer.buttons), 1)

    def test_compatibility_registration_activates_meta_and_defers_structure_directory(self):
        from web_ui import esmfold_backend, meta_backend

        viewer = FakeViewer()
        meta_backend.register(viewer)
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["meta"])
        self.assertEqual(len(viewer.buttons), 1)
        self.assertIn("import_metadata", viewer.web_action_handlers)

        with tempfile.TemporaryDirectory() as temp_dir:
            structures_dir = Path(temp_dir) / "not-created-during-registration"
            with mock.patch.object(
                esmfold_backend.cfg, "STRUCTURES_DIR", str(structures_dir)
            ):
                esmfold_backend.register_backend(
                    viewer.web_plugin_registry, viewer
                )
            self.assertFalse(structures_dir.exists())
            self.assertEqual(
                viewer.web_plugin_registry.static_routes["/structures/"],
                str(structures_dir.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
