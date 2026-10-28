"""Web plugin discovery, registration and declared pages (src/web_ui/Plugin_Manager.py).

A plugin's manifest (src/web_ui/plugins/<id>.py) declares its backend and its pages: where each is, what it is
called, its theme, the relaxations of the Content-Security-Policy it needs and the files of its texts. A backend
registers its actions, static routes, API routes and state. The web server, the translation update and the
Viewer's page opener read all of that, so none of them names a page: the tests below build a plugin that exists
only in a temporary tree and serve it with the real server, and scan the central code for the names of the bundled
pages, plugins and backends.
"""

import ast
import base64
import collections
import contextlib
import dataclasses
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities import Localization  # noqa: E402
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

    def add_console_button(self, *args, **kwargs):
        self.buttons.append((args, kwargs))

    def open_page(self, client):
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


class ManagerTests(unittest.TestCase):
    @staticmethod
    def _module(register):
        module = ModuleType("fake_backend")
        module.register_backend = register
        return module

    @staticmethod
    def _write_descriptor(folder, filename, plugin_id):
        manifest = {
            "api_version": 1,
            "id": plugin_id,
            "backend": f"fake_backends.{plugin_id}",
            "register": "register_backend",
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

    def test_bundled_registration_does_not_activate_console_buttons(self):
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

        agent_backend.activate(viewer)
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
                esmfold_backend, "STRUCTURES_DIRECTORY", str(structures_dir)
            ):
                esmfold_backend.register_backend(
                    viewer.web_plugin_registry, viewer
                )
            self.assertFalse(structures_dir.exists())
            self.assertEqual(
                viewer.web_plugin_registry.static_routes["/structures/"],
                str(structures_dir.resolve()),
            )


# ---------------------------------------------------------------------------
# The pages, API routes, policy, theme, texts and opener a manifest declares
# ---------------------------------------------------------------------------

BUNDLED_PLUGIN_DIR = SRC_DIR / "web_ui" / "plugins"
# The files that hold the three bundled pages' texts, and the catalog context of each, as the
# program had them in Page_Texts.PAGE_CONTEXTS before the manifests declared them.
EXPECTED_PAGE_CONTEXTS = {
    "web_ui/agent.html": "AgentPage",
    "resources/agent/attachments.js": "AgentPage",
    "web_ui/meta.html": "MetadataPage",
    "web_ui/esmfold.html": "EsmfoldPage",
    **{f"web_ui/esmfold_{name}.js": "EsmfoldPage" for name in (
        "shell", "layers", "superposition", "structures", "session", "color", "residue", "sequence", "tools",
        "image", "jobs")},
}


def page_entry(**changes):
    """One page as a manifest declares it, with the keys in `changes` replaced (None removes a key)."""
    entry = {
        "path": "/alpha.html", "label": "Alpha UI", "client": "alpha", "themed": True,
        "csp": {"connect-src": ["https:"]},
        "i18n": {"context": "AlphaPage", "files": ["web_ui/alpha.html", "web_ui/alpha_*.js"]},
    }
    for key, value in changes.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value
    return entry


def read_manifest(**manifest_changes):
    """Write a manifest with a page_entry as its only page, and read it back; a bad one raises WebPluginError."""
    manifest = {"api_version": 1, "id": "alpha", "backend": "alpha_backend", "register": "register_backend",
                "pages": [page_entry()]}
    manifest.update(manifest_changes)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "alpha.py"
        path.write_text(f"WEB_PLUGIN = {manifest!r}\n", encoding="utf-8")
        return Plugin_Manager.read_plugin_descriptor(str(path))


class BundledManifestTests(unittest.TestCase):
    """What the three bundled manifests declare of their pages: the program's own tables read from them."""

    def descriptors(self):
        descriptors, diagnostics = discover_plugin_descriptors(str(BUNDLED_PLUGIN_DIR))
        self.assertEqual(diagnostics, [])
        return {descriptor.plugin_id: descriptor for descriptor in descriptors}

    def test_each_plugin_declares_its_page(self):
        pages = {plugin_id: descriptor.pages for plugin_id, descriptor in self.descriptors().items()}
        self.assertEqual({plugin_id: [page.path for page in found] for plugin_id, found in pages.items()},
                         {"agent": ["/agent.html"], "esmfold": ["/esmfold.html"], "meta": ["/meta.html"]})
        agent, esmfold, meta = (pages[name][0] for name in ("agent", "esmfold", "meta"))
        self.assertEqual((agent.label, agent.client, agent.themed, agent.csp), ("Agent UI", "agent", True, ()))
        self.assertEqual((meta.label, meta.client, meta.themed, meta.csp), ("Metadata UI", "meta", True, ()))
        self.assertEqual((esmfold.label, esmfold.client, esmfold.themed), ("ESMFold Mol* UI", "esmfold", True))
        # The relaxations Mol* needs: the base policy plus these two, and no other page has any.
        self.assertEqual(esmfold.csp_additions, {"script-src": "'unsafe-eval'", "connect-src": "https: data:"})

    def test_the_declared_files_and_contexts_are_the_ones_the_program_had_before(self):
        # Page_Texts.PAGE_CONTEXTS is read from the manifests: the same 15 files in the same 3 contexts.
        from web_ui import Page_Texts

        self.assertEqual(Page_Texts.PAGE_CONTEXTS, EXPECTED_PAGE_CONTEXTS)
        contexts, diagnostics = Plugin_Manager.page_contexts(str(SRC_DIR))
        self.assertEqual((contexts, diagnostics), (EXPECTED_PAGE_CONTEXTS, []))

    def test_the_manifests_name_the_relaxations_the_server_allows(self):
        for descriptor in self.descriptors().values():
            for page in descriptor.pages:
                for directive, sources in page.csp:
                    self.assertTrue(set(sources) <= set(Plugin_Manager.ALLOWED_CSP_RELAXATIONS[directive]))

    def test_the_servers_allowlist_names_only_directives_of_its_base_policy(self):
        from web_ui import Web_Server

        self.assertTrue(set(Plugin_Manager.ALLOWED_CSP_RELAXATIONS) <= set(Web_Server.PAGE_POLICY))
        self.assertEqual(dict(Plugin_Manager.ALLOWED_CSP_RELAXATIONS),
                         {"script-src": ("'unsafe-eval'",), "connect-src": ("https:", "data:")})


class ManifestValidationTests(unittest.TestCase):
    """A manifest is checked as strictly for its pages as for its other keys."""

    def refused(self, *words, **manifest_changes):
        with self.assertRaises(WebPluginError) as caught:
            read_manifest(**manifest_changes)
        for word in words:
            self.assertIn(word, str(caught.exception))

    def refused_page(self, *words, **page_changes):
        self.refused(*words, pages=[page_entry(**page_changes)])

    def test_a_manifest_with_pages_is_read(self):
        descriptor = read_manifest()
        [page] = descriptor.pages
        self.assertEqual((page.plugin_id, page.path, page.label, page.client, page.themed),
                         ("alpha", "/alpha.html", "Alpha UI", "alpha", True))
        self.assertEqual(page.csp, (("connect-src", ("https:",)),))
        self.assertEqual((page.i18n_context, page.i18n_files), ("AlphaPage", ("web_ui/alpha.html", "web_ui/alpha_*.js")))
        self.assertEqual(str(page.title), "Alpha UI")
        # Without pages, as a plugin of an API route only.
        self.assertEqual(read_manifest(pages=[]).pages, ())
        manifest = {"api_version": 1, "id": "alpha", "backend": "b", "register": "r"}
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "alpha.py").write_text(f"WEB_PLUGIN = {manifest!r}\n", encoding="utf-8")
            self.assertEqual(Plugin_Manager.read_plugin_descriptor(str(Path(folder) / "alpha.py")).pages, ())

    def test_a_page_with_only_the_required_keys_takes_no_theme_policy_or_translation(self):
        [page] = read_manifest(pages=[page_entry(themed=None, csp=None, i18n=None)]).pages
        self.assertEqual((page.themed, page.csp, page.i18n_context, page.i18n_files), (False, (), None, ()))

    def test_unknown_and_missing_keys_are_refused(self):
        self.refused_page("unknown page field(s): theme", theme=True)
        self.refused_page("missing page field(s): client", client=None)
        self.refused_page("missing page field(s): label", label=None)
        self.refused_page("missing page field(s): path", path=None)
        self.refused_page("unknown i18n field(s): folder", i18n={"context": "A", "files": ["web_ui/a.html"], "folder": "x"})
        self.refused_page("missing i18n field(s): files", i18n={"context": "A"})
        self.refused(pages=[page_entry(), "alpha"])
        self.refused("pages must be a list", pages="alpha")
        self.refused("unknown manifest field(s): page", page=[])

    def test_the_old_activate_key_is_no_longer_part_of_the_manifest(self):
        # The manager never called it (a plugin's command does, through its backend's register); the key was dead.
        self.refused("unknown manifest field(s): activate", activate="activate")

    def test_the_path_must_be_an_html_page_not_under_a_reserved_prefix(self):
        for path in ("alpha.html", "/alpha.js", "/api/alpha.html", "/fonts/alpha.html", "/../alpha.html", "/a/../alpha.html",
                     "/a//alpha.html", "/alpha.html?x=1", "/alpha.html#x", "/a b.html", "/alpha\\beta.html", "/.html", 7, "",
                     "/./alpha.html", "/alpha.HTML"):
            with self.subTest(path=path):
                self.refused_page("page", path=path)
        for path in ("/alpha.html", "/alpha_two.html", "/alpha_resource/page.html", "/a-b/c.d.html"):
            with self.subTest(path=path):
                self.assertEqual(read_manifest(pages=[page_entry(path=path)]).pages[0].path, path)

    def test_the_label_is_plain_english_for_a_message(self):
        for label in ("", "   ", "{page} UI", "Alpha\nUI", "x" * 81, 3):
            with self.subTest(label=label):
                self.refused_page("label", label=label)

    def test_the_client_is_an_event_client_label(self):
        for client in ("", "Alpha", "a b", "a" * 65, "a.b", 3):
            with self.subTest(client=client):
                self.refused_page("client", client=client)

    def test_themed_is_a_boolean(self):
        for themed in ("yes", "true", 1, 0):
            with self.subTest(themed=themed):
                self.refused_page("themed", themed=themed)

    def test_a_relaxation_the_server_does_not_allow_is_refused(self):
        refused = [
            {"script-src": ["'unsafe-inline'"]},
            {"script-src": ["*"]},
            {"script-src": ["https:"]},
            {"connect-src": ["http:"]},
            {"connect-src": ["https:", "http://evil.example"]},
            {"connect-src": ["*"]},
            {"default-src": ["https:"]},
            {"frame-ancestors": ["https:"]},
            {"object-src": ["https:"]},
            {"style-src": ["'unsafe-eval'"]},
            {"script-src": []},
            {"script-src": "'unsafe-eval'"},
            {"connect-src": ["https:", "https:"]},
            ["connect-src"],
        ]
        for csp in refused:
            with self.subTest(csp=csp):
                self.refused_page("csp", csp=csp)
        # The allowed ones, together.
        allowed = {"script-src": ["'unsafe-eval'"], "connect-src": ["https:", "data:"]}
        self.assertEqual(read_manifest(pages=[page_entry(csp=allowed)]).pages[0].csp_additions,
                         {"script-src": "'unsafe-eval'", "connect-src": "https: data:"})
        # Whichever way a descriptor is made: the check is the descriptor's own.
        with self.assertRaises(WebPluginError):
            Plugin_Manager.WebPageDescriptor("alpha", "/alpha.html", "Alpha", "alpha", csp=(("script-src", ("'unsafe-inline'",)),))

    def test_the_text_context_and_files_are_checked(self):
        for i18n in (
            {"context": "Message", "files": ["web_ui/a.html"]},
            {"context": "a b", "files": ["web_ui/a.html"]},
            {"context": "", "files": ["web_ui/a.html"]},
            {"context": "A", "files": []},
            {"context": "A", "files": "web_ui/a.html"},
            {"context": "A", "files": ["../a.html"]},
            {"context": "A", "files": ["web_ui/../../a.html"]},
            {"context": "A", "files": ["/etc/a.html"]},
            {"context": "A", "files": ["C:/a.html"]},
            {"context": "A", "files": ["web_ui\\a.html"]},
            {"context": "A", "files": ["web_ui/a.html", "web_ui/a.html"]},
            {"context": "A", "files": [3]},
            ["A"],
        ):
            with self.subTest(i18n=i18n):
                self.refused_page("", i18n=i18n)

    def test_two_pages_of_a_plugin_may_not_share_a_path_or_a_client(self):
        self.refused("share a path", pages=[page_entry(), page_entry(client="beta")])
        self.refused("share a client", pages=[page_entry(), page_entry(path="/alpha_two.html")])

    def test_a_manifest_that_is_not_a_literal_is_refused_as_before(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "alpha.py"
            path.write_text("PAGES = []\nWEB_PLUGIN = {'api_version': 1, 'id': 'alpha', 'pages': PAGES}\n", encoding="utf-8")
            with self.assertRaises(WebPluginError):
                Plugin_Manager.read_plugin_descriptor(str(path))


class PageRegistryTests(unittest.TestCase):
    """register_page puts a page in the tables the server reads, and keeps a page its own plugin's."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.src = Path(folder.name)
        self.web = self.src / "web_ui"
        self.routes = self.src / "resources" / "alpha"
        for path in (self.web, self.routes):
            path.mkdir(parents=True)
        for name in ("alpha.html", "alpha_two.js", "alpha_one.js"):
            (self.web / name).write_text("x", encoding="utf-8")
        (self.routes / "page.html").write_text("x", encoding="utf-8")
        self.registry = WebPluginRegistry(object())

    def register(self, **changes):
        [page] = read_manifest(pages=[page_entry(**changes)]).pages
        self.registry.register_page(page, web_root=str(self.web), src_root=str(self.src))
        return page

    def test_a_page_of_the_web_folder_is_found_by_its_file_in_any_spelling(self):
        self.register()
        plain = self.web / "alpha.html"
        spellings = [plain, self.web / "plugins" / ".." / "alpha.html", self.web / "." / "alpha.html"]
        if Path("C:/").drive:
            spellings.append(Path(str(plain).upper()))
        for spelling in spellings:
            with self.subTest(spelling=str(spelling)):
                self.assertEqual(self.registry.csp_additions(spelling), {"connect-src": "https:"})
                self.assertTrue(self.registry.is_themed(spelling))
        for other in (self.web / "alpha_two.html", self.web / "sub" / "alpha.html", self.src / "alpha.html",
                      self.routes / "alpha.html", Path(tempfile.gettempdir()) / "alpha.html"):
            with self.subTest(other=str(other)):
                self.assertEqual(self.registry.csp_additions(other), {})
                self.assertFalse(self.registry.is_themed(other))

    def test_the_texts_of_a_page_are_found_by_file_with_the_pattern_expanded(self):
        self.register()
        for name in ("alpha.html", "alpha_one.js", "alpha_two.js"):
            self.assertEqual(self.registry.text_context(self.web / name), "AlphaPage", name)
        self.assertIsNone(self.registry.text_context(self.web / "other.js"))
        self.assertIsNone(self.registry.text_context(self.src / "alpha_one.js"))

    def test_a_page_is_found_by_its_client_and_one_nobody_declared_is_not(self):
        page = self.register()
        self.assertIs(self.registry.page("alpha"), page)
        with self.assertRaises(Plugin_Manager.WebPageUnavailable):
            self.registry.page("beta")

    def test_a_page_in_a_route_of_its_plugin_is_found_by_that_routes_file(self):
        self.registry.register_static_route("alpha", "/alpha_resource/", str(self.routes))
        self.register(path="/alpha_resource/page.html", i18n=None)
        self.assertTrue(self.registry.is_themed(self.routes / "page.html"))
        self.assertFalse(self.registry.is_themed(self.web / "page.html"))

    def test_a_page_in_the_route_of_another_plugin_is_refused(self):
        self.registry.register_static_route("beta", "/beta_resource/", str(self.routes))
        with self.assertRaisesRegex(WebPluginError, "route '/beta_resource/' of plugin 'beta'"):
            self.register(path="/beta_resource/page.html")
        self.assertEqual(self.registry.pages, {})

    def test_a_page_of_the_web_folder_is_named_for_its_plugin(self):
        # Else a plugin could claim another's page, or any page of the folder, for its relaxations.
        for path in ("/beta.html", "/alphabet.html", "/other/alpha.html"):
            with self.subTest(path=path), self.assertRaisesRegex(WebPluginError, "named for its plugin"):
                self.register(path=path, client="x")
        self.register(path="/alpha_two.html", client="two", i18n=None)
        self.register(path="/alpha/index.html", client="three", i18n=None)

    def test_a_page_may_be_declared_once_and_a_client_has_one_page(self):
        self.register()
        with self.assertRaisesRegex(WebPluginError, "client 'alpha'"):
            self.register(path="/alpha_two.html", i18n=None)
        with self.assertRaisesRegex(WebPluginError, "already registered"):
            self.register(client="two", i18n=None)

    def test_two_pages_cannot_be_one_file(self):
        # A route whose folder is the web folder serves the same file at another address.
        self.registry.register_static_route("alpha", "/alpha_files/", str(self.web))
        self.register()
        with self.assertRaisesRegex(WebPluginError, "already the page '/alpha.html'"):
            self.register(path="/alpha_files/alpha.html", client="two", i18n=None)

    def test_a_folder_with_glob_characters_in_its_name_is_read_as_a_path(self):
        odd = self.src / "odd[1]"
        (odd / "web_ui").mkdir(parents=True)
        for name in ("alpha.html", "alpha_one.js"):
            (odd / "web_ui" / name).write_text("x", encoding="utf-8")
        [page] = read_manifest().pages
        self.assertEqual(Plugin_Manager.i18n_files(page, str(odd)), ["web_ui/alpha.html", "web_ui/alpha_one.js"])

    def test_a_file_of_texts_belongs_to_one_page_and_a_context_to_one_plugin(self):
        self.register()
        with self.assertRaisesRegex(WebPluginError, "already holds the texts"):
            self.register(path="/alpha_two.html", client="two", i18n={"context": "AlphaPage", "files": ["web_ui/alpha.html"]})
        [other] = read_manifest(id="beta", pages=[page_entry(path="/beta.html", client="beta", i18n={
            "context": "AlphaPage", "files": ["web_ui/beta.js"]})]).pages
        (self.web / "beta.html").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(WebPluginError, "belongs to plugin 'alpha'"):
            self.registry.register_page(other, web_root=str(self.web), src_root=str(self.src))

    def test_a_pages_registration_is_rolled_back_with_its_plugin(self):
        # The manager registers the backend's capabilities and then the pages as one step: a page that is
        # refused (here one in another plugin's route) takes the plugin's earlier pages and actions with it.
        module = ModuleType("alpha_backend")
        module.register_backend = lambda registry, viewer: registry.register_action("alpha", "ping", lambda data: data)
        manifest = {"api_version": 1, "id": "alpha", "backend": "alpha_backend", "register": "register_backend",
                    "pages": [page_entry(i18n=None), page_entry(path="/beta_resource/x.html", client="two", i18n=None)]}
        plugins = self.web / "plugins"
        plugins.mkdir()
        (plugins / "alpha.py").write_text(f"WEB_PLUGIN = {manifest!r}\n", encoding="utf-8")
        manager = WebPluginManager(FakeViewer(), plugin_dir=str(plugins))
        manager.registry.register_static_route("beta", "/beta_resource/", str(self.routes))
        with mock.patch.object(Plugin_Manager.importlib, "import_module", return_value=module):
            manager.discover_and_register()
        self.assertEqual(manager.registry.registered_plugins, set())
        self.assertEqual((manager.registry.pages, manager.registry.actions), ({}, {}))
        self.assertEqual(manager.registry.csp_additions(self.web / "alpha.html"), {})
        self.assertTrue(any("route '/beta_resource/' of plugin 'beta'" in item.message for item in manager.diagnostics))


class ApiRouteRegistryTests(unittest.TestCase):
    """register_api_route: a plugin's own folder of /api/, and the checks the server runs for it."""

    def setUp(self):
        self.registry = WebPluginRegistry(object())

    def handler(self, request):
        return 200, {}

    def refused(self, plugin_id, path, *words, **options):
        with self.assertRaises(WebPluginError) as caught:
            self.registry.register_api_route(plugin_id, path, self.handler, **options)
        for word in words:
            self.assertIn(word, str(caught.exception))

    def test_a_plugin_registers_routes_only_under_its_own_folder_of_api(self):
        for path in ("/api/beta/x", "/api/alphabet/x", "/api/alpha", "/api/", "/api/action", "/api/mcp/v1/data", "/api/events",
                     "/x", "/fonts/alpha/", "/alpha/", "api/alpha/x", "/api/alpha/../beta/x", "/api/alpha/./x", "/api/alpha//x",
                     "/api/alpha/a b", "/api/alpha/x?y=1", "", None):
            with self.subTest(path=path):
                self.refused("alpha", path, "must be under /api/alpha/")
        self.registry.register_api_route("alpha", "/api/alpha/x", self.handler)
        self.registry.register_api_route("alpha", "/api/alpha/deep/folder/", self.handler)
        self.registry.register_api_route("alpha", "/api/alpha/", self.handler, methods=("GET",))
        self.assertEqual(sorted(path for _, path in self.registry.api_routes),
                         ["/api/alpha/", "/api/alpha/deep/folder/", "/api/alpha/x"])

    def test_the_servers_own_routes_cannot_be_taken_by_a_plugin_of_that_name(self):
        for plugin_id in sorted(Plugin_Manager.CORE_API_NAMES):
            with self.subTest(plugin=plugin_id):
                self.refused(plugin_id, f"/api/{plugin_id}/x", "is the server's")
        self.assertEqual(Plugin_Manager.CORE_API_NAMES, {"action", "events", "mcp"})

    def test_a_plugin_cannot_register_for_another_while_it_registers(self):
        self.registry._active_registration_owner = "beta"
        self.refused("alpha", "/api/alpha/x", "attempted to register")

    def test_the_methods_and_the_bodys_limits_are_checked(self):
        self.refused("alpha", "/api/alpha/x", "needs max_body", methods=("POST",))
        self.refused("alpha", "/api/alpha/x", "needs max_body", methods=("GET", "POST"), max_body=0)
        self.refused("alpha", "/api/alpha/x", "needs max_body", methods=("POST",), max_body=True)
        self.refused("alpha", "/api/alpha/x", "takes no body", max_body=10)
        self.refused("alpha", "/api/alpha/x", "takes no body", content_types=("application/json",))
        self.refused("alpha", "/api/alpha/x", "GET and/or POST", methods=("DELETE",))
        self.refused("alpha", "/api/alpha/x", "GET and/or POST", methods=("GET", "GET"))
        self.refused("alpha", "/api/alpha/x", "GET and/or POST", methods=())
        self.refused("alpha", "/api/alpha/x", "lowercase media types", methods=("POST",), max_body=5, content_types=("JSON",))
        with self.assertRaises(WebPluginError):
            self.registry.register_api_route("alpha", "/api/alpha/x", "not callable")
        self.registry.register_api_route("alpha", "/api/alpha/post", self.handler, methods="POST", max_body=5)
        route = self.registry.api_routes[("POST", "/api/alpha/post")]
        self.assertEqual((route.max_body, route.content_types), (5, None))

    def test_a_route_registered_again_by_its_plugin_replaces_it_as_backends_register_again(self):
        first, second = (lambda request: (200, 1)), (lambda request: (200, 2))
        self.registry.register_api_route("alpha", "/api/alpha/x", first)
        self.registry.register_api_route("alpha", "/api/alpha/x", second)
        self.assertIs(self.registry.api_routes[("GET", "/api/alpha/x")].handler, second)
        self.registry.api_routes[("GET", "/api/alpha/x")] = Plugin_Manager.ApiRoute("beta", "/api/alpha/x", "GET", first)
        self.refused("alpha", "/api/alpha/x", "already registered by plugin 'beta'")

    def test_the_most_specific_route_answers(self):
        for path in ("/api/alpha/", "/api/alpha/deep/", "/api/alpha/x"):
            self.registry.register_api_route("alpha", path, lambda request, path=path: (200, path))
        match = lambda path, method="GET": Plugin_Manager.match_api_route(self.registry.api_routes, method, path)
        self.assertEqual(match("/api/alpha/x").path, "/api/alpha/x")
        self.assertEqual(match("/api/alpha/x/y").path, "/api/alpha/")
        self.assertEqual(match("/api/alpha/deep/z/w").path, "/api/alpha/deep/")
        self.assertEqual(match("/api/alpha/deeper").path, "/api/alpha/")
        self.assertIsNone(match("/api/alpha"))
        self.assertIsNone(match("/api/beta/x"))
        self.assertIsNone(match("/api/alpha/x", "POST"))

    def test_a_failed_registration_takes_back_its_routes(self):
        modules = {"alpha_backend": ModuleType("alpha_backend")}

        def register(registry, viewer):
            registry.register_api_route("alpha", "/api/alpha/x", self.handler)
            raise RuntimeError("intentional failure")

        modules["alpha_backend"].register_backend = register
        with tempfile.TemporaryDirectory() as folder:
            manifest = {"api_version": 1, "id": "alpha", "backend": "alpha_backend", "register": "register_backend"}
            (Path(folder) / "alpha.py").write_text(f"WEB_PLUGIN = {manifest!r}\n", encoding="utf-8")
            manager = WebPluginManager(FakeViewer(), plugin_dir=folder)
            with mock.patch.object(Plugin_Manager.importlib, "import_module", side_effect=lambda name: modules[name]):
                manager.discover_and_register()
        self.assertEqual(manager.registry.api_routes, {})


# ---------------------------------------------------------------------------
# A plugin that exists only in a temporary tree, served by the real web server
# ---------------------------------------------------------------------------

DUMMY_PAGE = (
    '<!DOCTYPE html>\n<html lang="en">\n<head><title data-i18n>Dummy title</title></head>\n'
    '<body><p data-i18n>Hello dummy</p><script>show(t("Dummy shown"));</script></body>\n</html>\n'
)
DUMMY_SCRIPT = 'show(t("Dummy script text"));\n'
INLINE_SCRIPT = re.compile(rb"<script>(.*?)</script>", re.DOTALL)


def sha256_source(text):
    return "'sha256-%s'" % base64.b64encode(hashlib.sha256(text).digest()).decode()


def directives(policy):
    return {name: sources for name, *sources in (part.split() for part in policy.split(";"))}


class DummyPluginTests(unittest.TestCase):
    """What a plugin declares is all the web server needs: nothing in the program names the plugin.

    The plugin is a manifest, a backend and a few files in a temporary tree: a page
    (themed, with a relaxed policy and translated texts) and its script, a page that
    declares nothing, and a page of the same folder that no manifest names.
    """

    MANIFEST = {
        "api_version": 1, "id": "dummy", "backend": "dummy_backend", "register": "register_backend",
        "pages": [
            {"path": "/dummy_resource/page.html", "label": "Dummy UI", "client": "dummy", "themed": True,
             "csp": {"script-src": ["'unsafe-eval'"], "connect-src": ["https:"]},
             "i18n": {"context": "DummyPage", "files": ["resources/dummy/page.*"]}},
            {"path": "/dummy_resource/plain.html", "label": "Plain UI", "client": "dummy_plain"},
        ],
    }
    BODY_LIMIT = 32

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        from EMAPSSN_Viewer import MainViewer

        cls.app = QApplication.instance() or QApplication([])
        cls.MainViewer = MainViewer

    def setUp(self):
        from desktop.Viewer_Inspection import ViewerInspectionService
        from web_ui import Web_Server

        self.Web_Server = Web_Server
        sessions = tempfile.TemporaryDirectory()
        tree = tempfile.TemporaryDirectory()
        self.addCleanup(sessions.cleanup)
        self.addCleanup(tree.cleanup)
        environment = mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": sessions.name})
        environment.start()
        self.addCleanup(environment.stop)
        self.root = Path(tree.name)
        self.plugins = self.root / "web_ui" / "plugins"
        self.files = self.root / "resources" / "dummy"
        for folder in (self.plugins, self.files):
            folder.mkdir(parents=True)
        for name, text in (("page.html", DUMMY_PAGE), ("page.js", DUMMY_SCRIPT), ("plain.html", DUMMY_PAGE),
                           ("other.html", DUMMY_PAGE), ("notes.py", "SECRET = 1\n")):
            (self.files / name).write_text(text, encoding="utf-8", newline="\n")
        self.write_manifest("dummy", self.MANIFEST)

        self.viewer = SimpleNamespace(
            n_nodes=2, full_headers=["a", "b"], metadata={}, edges=[], visible_mask=[True, True], selected_indices=[],
            group_labels=[[], []], cluster_labels=None, pings=[])
        self.registry = self.viewer.web_plugin_registry = WebPluginRegistry(self.viewer)
        self.viewer.web_action_handlers = self.registry.actions
        self.viewer.viewer_inspection = ViewerInspectionService(self.viewer)
        self.viewer.communicator = Web_Server.QtCommunicator(self.viewer)
        self.viewer.handle_web_action = types.MethodType(self.MainViewer.handle_web_action, self.viewer)
        self.viewer.get_initial_web_state = lambda: {"rows": []}

        self.backend = ModuleType("dummy_backend")
        self.backend.register_backend = self.register_backend
        # The backends the plugin manager imports, by module name.
        self.modules = {"dummy_backend": self.backend}
        self.manager = WebPluginManager(self.viewer, plugin_dir=str(self.plugins))
        self.diagnostics_printed = io.StringIO()

    def write_manifest(self, name, manifest):
        (self.plugins / f"{name}.py").write_text(f"WEB_PLUGIN = {manifest!r}\n", encoding="utf-8")

    def register_backend(self, registry, viewer):
        registry.register_static_route("dummy", "/dummy_resource/", str(self.files))
        registry.register_action("dummy", "ping", lambda data: viewer.pings.append(data["value"]))
        registry.register_api_route(
            "dummy", "/api/dummy/echo/", self.echo, methods=("GET", "POST"), max_body=self.BODY_LIMIT,
            content_types=("application/json",), size_error="The dummy takes at most 32 bytes.")

    @staticmethod
    def echo(request):
        if request.body == b'"bad"':
            raise ValueError("Bad dummy value.")
        return 200, {"method": request.method, "path": request.path, "query": request.query,
                     "body": request.body.decode("utf-8")}

    def start(self):
        """Register the plugins of the temporary tree, as the Viewer does, and start the real server."""
        with mock.patch.object(Plugin_Manager.importlib, "import_module", side_effect=lambda name: self.modules[name]), \
                contextlib.redirect_stdout(self.diagnostics_printed):
            self.manager.discover_and_register()
        self.server = self.Web_Server.start_server(self.viewer, preferred_port=0)
        self.addCleanup(self.Web_Server.stop_server, self.server)
        self.port = self.server.server_address[1]
        self.same_origin = {"Origin": f"http://127.0.0.1:{self.port}", "Sec-Fetch-Site": "same-origin"}
        return self

    def exchange(self, method, path, headers=None, body=b"", content_length=None):
        """(status, headers, body bytes) of a request to the server, whose Host is its own unless headers name one.

        content_length, a number of bytes the body is said to have, is sent in place of the body's own length:
        the server answers a body it will not read before the body is sent, which a socket that closes with
        the body unread would turn into a reset on Windows.
        """
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            for name, value in {"Host": f"127.0.0.1:{self.port}", **(headers or {})}.items():
                connection.putheader(name, value)
            if content_length is not None:
                connection.putheader("Content-Length", str(content_length))
            elif body:
                connection.putheader("Content-Length", str(len(body)))
            connection.endheaders()
            if body:
                connection.send(body)
            response = connection.getresponse()
            return response.status, response.headers, response.read()
        finally:
            connection.close()

    def policy(self, path):
        status, headers, _ = self.exchange("GET", path)
        self.assertEqual(status, 200, path)
        return directives(headers["Content-Security-Policy"])

    def served_in_another_language(self):
        """The server with a language whose catalog is a function of the context and the text."""
        for target, replacement in (
            ("installed_language", mock.Mock(return_value="xx")),
            ("translate", lambda context, text: f"[{context}] {text}"),
            ("language_web_font_css", mock.Mock(return_value="")),
        ):
            patch = mock.patch.object(self.Web_Server, target, replacement)
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_plugin_registered_with_its_pages_and_nothing_else_was_told_of_it(self):
        self.start()
        self.assertEqual(self.diagnostics_printed.getvalue(), "")
        self.assertEqual(self.registry.registered_plugins, {"dummy"})
        self.assertEqual(sorted(self.registry.pages), ["dummy", "dummy_plain"])
        self.assertEqual(sorted(self.registry.static_routes), ["/dummy_resource/", "/fonts/"])

    def test_the_page_is_served_at_its_path_with_only_its_declared_policy(self):
        self.start()
        base = {name: sources.split() for name, sources in self.Web_Server.PAGE_POLICY.items()}
        for path in ("/dummy_resource/page.html", "/dummy_resource/page.html?x=1", "/dummy_resource/./page.html"):
            with self.subTest(path=path):
                status, headers, body = self.exchange("GET", path)
                self.assertEqual(status, 200)
                self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
                policy = directives(headers["Content-Security-Policy"])
                [script] = INLINE_SCRIPT.findall(body)
                self.assertEqual(policy["script-src"], ["'self'", "'unsafe-eval'", sha256_source(script)])
                self.assertEqual(policy["connect-src"], ["'self'", "https:"])
                self.assertEqual({name: value for name, value in policy.items() if name not in ("script-src", "connect-src")},
                                 {name: value for name, value in base.items() if name not in ("script-src", "connect-src")})

    def test_a_page_that_declares_nothing_gets_the_base_policy_even_beside_one_that_does(self):
        self.start()
        base = {name: sources.split() for name, sources in self.Web_Server.PAGE_POLICY.items()}
        for path in ("/dummy_resource/plain.html", "/dummy_resource/other.html"):
            with self.subTest(path=path):
                status, headers, body = self.exchange("GET", path)
                self.assertEqual(status, 200)
                policy = directives(headers["Content-Security-Policy"])
                [script] = INLINE_SCRIPT.findall(body)
                self.assertEqual(policy, {**base, "script-src": ["'self'", sha256_source(script)]})

    def test_a_page_of_the_program_the_plugin_does_not_declare_does_not_get_the_policy(self):
        # The bundled Fold View page is served from the web folder, and this plugin's server declares none
        # of the bundled plugins: the relaxation follows the declaration, not the name or the folder.
        self.start()
        for path in ("/esmfold.html", "/agent.html", "/meta.html"):
            with self.subTest(path=path):
                policy = self.policy(path)
                self.assertNotIn("'unsafe-eval'", policy["script-src"])
                self.assertEqual(policy["connect-src"], ["'self'"])

    def test_a_declared_pages_own_file_alone_has_the_relaxation(self):
        # The same bytes at the paths of the other pages, and at the declared path in another folder, are not it.
        self.start()
        other_folder = self.root / "elsewhere"
        other_folder.mkdir()
        (other_folder / "page.html").write_text(DUMMY_PAGE, encoding="utf-8")
        self.registry.register_static_route("dummy", "/elsewhere/", str(other_folder))
        self.assertNotIn("'unsafe-eval'", self.policy("/elsewhere/page.html")["script-src"])
        self.assertIn("'unsafe-eval'", self.policy("/dummy_resource/page.html")["script-src"])

    def test_the_theme_is_injected_only_into_a_page_that_declares_it(self):
        from desktop.Studio_Theme import web_theme_css

        self.start()
        css = web_theme_css().encode("utf-8")
        for path, themed in (("/dummy_resource/page.html", True), ("/dummy_resource/plain.html", False),
                             ("/dummy_resource/other.html", False), ("/esmfold.html", False)):
            with self.subTest(path=path):
                status, _, body = self.exchange("GET", path)
                self.assertEqual(status, 200)
                self.assertEqual(body.count(css), 1 if themed else 0)
                if themed:
                    head, end, _ = body.partition(b"</head>")
                    self.assertTrue(end and head.endswith(b"<style>\n" + css + b"\n</style>\n"))
        # The script is not a page: it takes no theme either.
        self.assertEqual(self.exchange("GET", "/dummy_resource/page.js")[2].count(css), 0)

    def test_the_marked_texts_are_translated_through_the_declared_context(self):
        self.served_in_another_language()
        self.start()
        _, _, page = self.exchange("GET", "/dummy_resource/page.html")
        text = page.decode("utf-8")
        for shown in ("<title data-i18n>[DummyPage] Dummy title</title>", "<p data-i18n>[DummyPage] Hello dummy</p>",
                      'show(t("[DummyPage] Dummy shown"));', '<html lang="xx">'):
            self.assertIn(shown, text)
        # A script of the declared files, matched by the pattern, comes translated too.
        self.assertEqual(self.exchange("GET", "/dummy_resource/page.js")[2].decode("utf-8"),
                         'show(t("[DummyPage] Dummy script text"));\n')
        # The same markup in a file no manifest names stays English, and so does a declared page without texts.
        for path in ("/dummy_resource/plain.html", "/dummy_resource/other.html"):
            with self.subTest(path=path):
                self.assertEqual(self.exchange("GET", path)[2].decode("utf-8").count("DummyPage"), 0)
                self.assertIn("<p data-i18n>Hello dummy</p>", self.exchange("GET", path)[2].decode("utf-8"))
        # The program's own pages are not this plugin's: no context of its is theirs.
        self.assertEqual(self.exchange("GET", "/agent.html")[2].count(b"[DummyPage]"), 0)

    def test_the_plugins_action_works(self):
        self.start()
        status, _, body = self.exchange(
            "POST", "/api/action", {**self.same_origin, "Content-Type": "application/json"},
            json.dumps({"action": "ping", "value": 7}).encode())
        self.assertEqual((status, json.loads(body)), (200, {"status": "ok"}))
        for _ in range(50):
            self.app.processEvents()
        self.assertEqual(self.viewer.pings, [7])

    def test_the_plugins_api_route_answers(self):
        self.start()
        status, headers, body = self.exchange("GET", "/api/dummy/echo/a/b?x=1", self.same_origin)
        self.assertEqual((status, json.loads(body)), (200, {"method": "GET", "path": "/api/dummy/echo/a/b", "query": "x=1", "body": ""}))
        self.assertEqual((headers["Content-Type"], headers["Cache-Control"]), ("application/json; charset=utf-8", "no-store"))
        json_headers = {**self.same_origin, "Content-Type": "application/json; charset=utf-8"}
        status, _, body = self.exchange("POST", "/api/dummy/echo/", json_headers, b'{"a": 1}')
        self.assertEqual((status, json.loads(body)["body"]), (200, '{"a": 1}'))
        # The handler's own refusal comes as a 400 with its text.
        status, _, body = self.exchange("POST", "/api/dummy/echo/", json_headers, b'"bad"')
        self.assertEqual((status, json.loads(body)), (400, {"error": "Bad dummy value."}))
        # An address that no route has is not answered by it.
        self.assertEqual(self.exchange("GET", "/api/dummy", self.same_origin)[0], 404)
        self.assertEqual(self.exchange("GET", "/api/other/echo/x", self.same_origin)[0], 404)

    def test_the_server_checks_a_posts_body_and_content_type_before_the_handler_sees_it(self):
        self.start()
        seen = []
        route = self.registry.api_routes[("POST", "/api/dummy/echo/")]
        self.registry.api_routes[("POST", "/api/dummy/echo/")] = dataclasses.replace(
            route, handler=lambda request: seen.append(request) or (200, {}))
        json_headers = {**self.same_origin, "Content-Type": "application/json"}
        too_large = self.exchange("POST", "/api/dummy/echo/", json_headers, content_length=self.BODY_LIMIT + 1)
        self.assertEqual((too_large[0], json.loads(too_large[2]), too_large[1]["Connection"]),
                         (413, {"error": "The dummy takes at most 32 bytes."}, "close"))
        empty = self.exchange("POST", "/api/dummy/echo/", json_headers)
        self.assertEqual((empty[0], json.loads(empty[2])), (413, {"error": "The dummy takes at most 32 bytes."}))
        wrong_type = self.exchange("POST", "/api/dummy/echo/", {**self.same_origin, "Content-Type": "text/plain"}, b"{}")
        self.assertEqual((wrong_type[0], json.loads(wrong_type[2])),
                         (415, {"error": "Requests must use Content-Type: application/json"}))
        self.assertEqual(seen, [])
        self.assertEqual(self.exchange("POST", "/api/dummy/echo/", json_headers, b"x" * self.BODY_LIMIT)[0], 200)
        [request] = seen
        self.assertEqual((request.method, request.path, request.body), ("POST", "/api/dummy/echo/", b"x" * self.BODY_LIMIT))

    def test_a_foreign_origin_and_host_are_refused_on_every_route_of_the_plugin(self):
        self.start()
        cases = [("GET", "/dummy_resource/page.html", b""), ("GET", "/dummy_resource/page.js", b""),
                 ("GET", "/dummy_resource/other.html", b""), ("GET", "/api/dummy/echo/x", b""),
                 ("POST", "/api/dummy/echo/", b"{}"), ("POST", "/api/action", b'{"action": "ping", "value": 1}')]
        refusals = {
            "origin": ({"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}, "Cross-origin requests are not accepted."),
            "site": ({"Sec-Fetch-Site": "cross-site"}, "Cross-site requests are not accepted."),
            "host": ({"Host": "evil.example"}, "Requests must address this Viewer as 127.0.0.1 or localhost."),
        }
        for method, path, body in cases:
            for kind, (headers, reason) in refusals.items():
                with self.subTest(method=method, path=path, kind=kind):
                    status, _, payload = self.exchange(method, path, {**headers, "Content-Type": "application/json"}, body)
                    self.assertEqual((status, json.loads(payload)), (403, {"error": reason}))
        for _ in range(20):
            self.app.processEvents()
        self.assertEqual(self.viewer.pings, [])

    def test_a_path_that_leaves_the_plugins_folder_is_refused_and_its_python_is_not_served(self):
        self.start()
        for path in ("/dummy_resource/../web_ui/plugins/dummy.py", "/dummy_resource/sub/../../resources/dummy/notes.py",
                     "/dummy_resource/..", "/dummy_resource/../"):
            with self.subTest(path=path):
                self.assertEqual(self.exchange("GET", path)[0], 403)
        for path in ("/dummy_resource/notes.py", "/dummy_resource/NOTES.PY", "/dummy_resource/missing.html"):
            with self.subTest(path=path):
                self.assertEqual(self.exchange("GET", path)[0], 404)

    def test_the_opener_uses_the_declared_path_label_and_client(self):
        self.start()
        viewer = SimpleNamespace(_open_web_ui=mock.Mock(return_value=True), web_plugin_registry=self.registry)
        for client, path, label in (("dummy", "/dummy_resource/page.html", "Dummy UI"),
                                    ("dummy_plain", "/dummy_resource/plain.html", "Plain UI")):
            for options, dialog in (({}, True), ({"show_existing_dialog": False}, False)):
                with self.subTest(client=client, options=options):
                    viewer._open_web_ui.reset_mock()
                    self.assertTrue(self.MainViewer.open_page(viewer, client, **options))
                    viewer._open_web_ui.assert_called_once_with(path, mock.ANY, client, show_existing_dialog=dialog)
                    self.assertEqual(str(viewer._open_web_ui.call_args.args[1]), label)
        # The label is a Message: translated when it is shown, from the catalog of Message texts.
        message = viewer._open_web_ui.call_args.args[1]
        translator = Localization.set_translator(lambda template, n: f"<{template}>")
        self.addCleanup(Localization.set_translator, translator)
        self.assertEqual(Localization.display_text(message), "<Plain UI>")

    def test_a_relaxation_the_server_does_not_allow_is_refused_and_the_plugin_is_not_registered(self):
        greedy = {"api_version": 1, "id": "greedy", "backend": "greedy_backend", "register": "register_backend",
                  "pages": [{"path": "/dummy_resource/greedy.html", "label": "Greedy UI", "client": "greedy",
                             "csp": {"script-src": ["'unsafe-inline'"]}}]}
        self.write_manifest("greedy", greedy)
        (self.files / "greedy.html").write_text(DUMMY_PAGE, encoding="utf-8")
        self.start()
        self.assertEqual(self.registry.registered_plugins, {"dummy"})
        self.assertNotIn("greedy", self.registry.pages)
        printed = self.diagnostics_printed.getvalue()
        self.assertIn("Web plugin <unknown> (greedy.py)", printed)
        self.assertIn("csp may not add 'unsafe-inline' to 'script-src'", printed)
        self.assertNotIn("'unsafe-inline'", self.policy("/dummy_resource/greedy.html")["script-src"])

    def test_a_plugin_cannot_declare_a_page_in_another_plugins_folder_or_an_api_route_outside_its_own(self):
        intruder = {"api_version": 1, "id": "intruder", "backend": "intruder_backend", "register": "register_backend",
                    "pages": [{"path": "/dummy_resource/other.html", "label": "Intruder UI", "client": "intruder",
                               "csp": {"connect-src": ["https:", "data:"]}}]}
        self.write_manifest("intruder", intruder)
        self.modules["intruder_backend"] = ModuleType("intruder_backend")
        self.modules["intruder_backend"].register_backend = lambda registry, viewer: None
        self.start()
        self.assertEqual(self.registry.registered_plugins, {"dummy"})
        self.assertIn("route '/dummy_resource/' of plugin 'dummy'", self.diagnostics_printed.getvalue())
        self.assertEqual(self.registry.csp_additions(self.files / "page.html"),
                         {"script-src": "'unsafe-eval'", "connect-src": "https:"})
        self.assertEqual(self.registry.csp_additions(self.files / "other.html"), {})
        for plugin_id, path in (("dummy", "/api/action/x"), ("dummy", "/api/mcp/v1/data"), ("dummy", "/api/other/x"),
                                ("dummy", "/api/events"), ("mcp", "/api/mcp/x")):
            with self.subTest(path=path), self.assertRaises(WebPluginError):
                self.registry.register_api_route(plugin_id, path, lambda request: (200, {}))


# ---------------------------------------------------------------------------
# The central web code names no page, plugin or backend; every page is declared
# ---------------------------------------------------------------------------

WEB_DIR = SRC_DIR / "web_ui"
# The files of web_ui that are the web layer itself, which serve, register, open and translate the plugins'
# pages; every other .py there is a plugin's backend or a helper of one.
CENTRAL_MODULES = frozenset({"Web_Server", "Plugin_Manager", "Browser_Page", "Page_Texts"})
# The methods of the Viewer that are its web layer (the metadata data model is the Viewer's own).
VIEWER_WEB_METHODS = frozenset({
    "open_page", "_open_web_ui", "get_web_url", "start_web_server", "ensure_web_server", "shutdown_web_server",
    "broadcast_event", "handle_web_action", "get_initial_web_state",
})
CENTRAL_WEB_FILES = ("src/web_ui/Web_Server.py", "src/web_ui/Plugin_Manager.py", "src/web_ui/Browser_Page.py",
                     "src/web_ui/Page_Texts.py")
VIEWER_FILE = "src/EMAPSSN_Viewer.py"
# What a central file may hold that is a page's, plugin's or backend's name as far as a scan can tell, by
# (scope, kind, name): how many times, and why that is not a page, a plugin or a backend named.
ALLOWED = {
    "src/web_ui/Page_Texts.py": {
        ("<module>", "string", "meta"): (1, "HTML's <meta> element, which has no end tag: the name of a tag, not of the plugin"),
    },
    "src/web_ui/Web_Server.py": {},
    "src/web_ui/Plugin_Manager.py": {},
    "src/web_ui/Browser_Page.py": {},
}


def declared_page_names():
    """The names the plugins' own files give: ids, backends, clients, labels, contexts, pages and their files.

    Read from disk, so a plugin added later is covered without a change here. Returns (words, paths):
    words a string constant may equal, paths that a string constant may contain.
    """
    descriptors, diagnostics = discover_plugin_descriptors(str(BUNDLED_PLUGIN_DIR))
    assert not diagnostics, diagnostics
    words, paths = set(), set()
    for descriptor in descriptors:
        words |= {descriptor.plugin_id, descriptor.backend, descriptor.backend.split(".")[-1], Path(descriptor.source_path).stem}
        for page in descriptor.pages:
            words |= {page.client, page.label, page.i18n_context} - {None}
            paths |= {page.path, page.path.lstrip("/")}
    modules = {path.stem for path in WEB_DIR.glob("*.py") if path.stem not in CENTRAL_MODULES and path.stem != "__init__"}
    words |= modules | {f"web_ui.{name}" for name in modules}
    paths |= {path.name for path in WEB_DIR.glob("*.html")} | {path.name for path in WEB_DIR.glob("*.js")}
    return words, paths, modules


def page_words_in(source, words, paths, modules):
    """Counter of (enclosing scope, 'string' | 'import' | 'name', word) for the code of a module's source.

    A string constant equal to a plugin's id, backend, client, label or catalog context, or holding the path of a
    page or a file of the web folder; an import of a plugin's module; a use of an imported plugin module's name.
    Docstrings and comments are not counted.
    """
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))

    def scope(node):
        parts = []
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                parts.append(node.name)
        return ".".join(reversed(parts)) or "<module>"

    found = collections.Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            if node.value in words:
                found[(scope(node), "string", node.value)] += 1
            else:
                found.update((scope(node), "string", path) for path in paths if path in node.value)
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")
            if module == ["web_ui"]:
                found.update((scope(node), "import", alias.name) for alias in node.names if alias.name in modules)
            elif module[0] == "web_ui" and len(module) > 1 and module[1] in modules:
                found[(scope(node), "import", module[1])] += 1
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "web_ui" and len(parts) > 1 and parts[1] in modules:
                    found[(scope(node), "import", parts[1])] += 1
        elif isinstance(node, ast.Name) and node.id in modules:
            found[(scope(node), "name", node.id)] += 1
    return found


class CentralCodeNamesNoPageTests(unittest.TestCase):
    """The web server, the plugin manager, the page opener, the page texts and the Viewer's web code name no page.

    A plugin declares its pages, routes and texts (web_ui/plugins/<id>.py); the central code reads what
    they declare, and a new plugin needs no change in it.
    """

    def test_the_scanner_finds_a_page_named_in_code(self):
        # A positive control: without it a scanner that finds nothing would pass every file.
        words = {"agent", "Agent UI", "AgentPage", "esmfold_backend", "web_ui.meta_backend"}
        paths = {"/agent.html", "agent.html", "esmfold_shell.js"}
        modules = {"agent_backend", "esmfold_backend", "meta_backend"}
        source = ("from web_ui import agent_backend\n"
                  "import web_ui.esmfold_backend\n"
                  "from web_ui.meta_backend import handle\n"
                  "from web_ui import Plugin_Manager\n"
                  "'''a docstring about the agent'''\n"
                  "def serve(path):\n"
                  "    '''A docstring that says agent.html.'''\n"
                  "    if path == '/agent.html' or path in ('agent', 'Agent UI', 'AgentPage', 'sidebar'):\n"
                  "        return agent_backend.go('web_ui/esmfold_shell.js', 'web_ui.meta_backend')\n"
                  "    return 'a sentence about the agent'\n")
        self.assertEqual(page_words_in(source, words, paths, modules), collections.Counter({
            ("<module>", "import", "agent_backend"): 1, ("<module>", "import", "esmfold_backend"): 1,
            ("<module>", "import", "meta_backend"): 1,
            ("serve", "string", "/agent.html"): 1, ("serve", "string", "agent.html"): 1,  # a string holds both paths
            ("serve", "string", "agent"): 1, ("serve", "string", "Agent UI"): 1, ("serve", "string", "AgentPage"): 1,
            ("serve", "string", "esmfold_shell.js"): 1, ("serve", "string", "web_ui.meta_backend"): 1,
            ("serve", "name", "agent_backend"): 1,
        }))

    def test_the_names_are_read_from_the_plugins_and_the_web_folder_on_disk(self):
        words, paths, modules = declared_page_names()
        self.assertTrue({"agent", "esmfold", "meta", "AgentPage", "Agent UI", "agent_backend", "web_ui.esmfold_backend"} <= words)
        self.assertTrue({"/agent.html", "meta.html", "esmfold.html", "esmfold_shell.js", "page_text.js"} <= paths)
        self.assertTrue({"agent_backend", "agent_images", "esmfold_backend", "meta_backend", "residue_layers"} <= modules)
        self.assertFalse(modules & CENTRAL_MODULES)

    def test_no_central_web_file_names_a_page_plugin_or_backend(self):
        words, paths, modules = declared_page_names()
        for relative in CENTRAL_WEB_FILES:
            source = (SRC_DIR.parent / relative).read_text(encoding="utf-8")
            found = page_words_in(source, words, paths, modules)
            expected = collections.Counter({key: count for key, (count, _reason) in ALLOWED[relative].items()})
            with self.subTest(file=relative):
                self.assertFalse(found - expected, f"{relative} names a page, plugin or backend in code: {dict(found - expected)}. "
                                 "A plugin declares them in its manifest (web_ui/plugins), and this code reads that.")
                self.assertFalse(expected - found, f"{relative} no longer holds {dict(expected - found)}: remove it from ALLOWED.")

    def test_the_viewers_web_code_names_no_page_plugin_or_backend(self):
        import textwrap

        words, paths, modules = declared_page_names()
        source = (SRC_DIR.parent / VIEWER_FILE).read_text(encoding="utf-8")
        found_methods = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name in VIEWER_WEB_METHODS | {"_web_clients_connected"}:
                found_methods.add(node.name)
                text = textwrap.dedent(" " * node.col_offset + ast.get_source_segment(source, node))
                with self.subTest(function=node.name):
                    self.assertEqual(page_words_in(text, words, paths, modules), collections.Counter())
        self.assertEqual(found_methods, VIEWER_WEB_METHODS | {"_web_clients_connected"})

    def test_the_central_files_are_the_ones_that_serve_open_and_translate(self):
        # A file that moved or was renamed would otherwise stop being checked.
        for relative in CENTRAL_WEB_FILES:
            self.assertTrue((SRC_DIR.parent / relative).is_file(), relative)
        self.assertEqual(set(ALLOWED), set(CENTRAL_WEB_FILES))
        self.assertEqual({path.stem for path in WEB_DIR.glob("*.py")} & CENTRAL_MODULES, CENTRAL_MODULES)


class EveryPageIsDeclaredTests(unittest.TestCase):
    """Every page of the web folder and every script with texts belongs to exactly one plugin's manifest."""

    SCRIPT = re.compile(r'<script\b[^>]*\ssrc\s*=\s*"([^"]+)"', re.IGNORECASE)
    # The libraries the pages load as they are: their own minified t() is not the pages' t(), so their text
    # is no page's.
    VENDORED = {
        "resources/agent/marked.umd.js": "the Markdown renderer of the Agent page",
        "resources/meta/tabulator.min.js": "the table library of the Metadata page",
        "resources/esmfold/molstar.js": "the Mol* bundle of the Fold View page",
    }

    @classmethod
    def setUpClass(cls):
        cls.descriptors, diagnostics = discover_plugin_descriptors(str(BUNDLED_PLUGIN_DIR))
        assert not diagnostics, diagnostics
        cls.pages = [page for descriptor in cls.descriptors for page in descriptor.pages]

    def test_every_html_page_of_the_web_folder_is_one_pages_file(self):
        declared = [page.path for page in self.pages]
        self.assertEqual(len(declared), len(set(declared)))
        found = ["/" + path.relative_to(WEB_DIR).as_posix() for path in sorted(WEB_DIR.rglob("*.html"))]
        self.assertTrue(found)
        self.assertEqual(sorted(found), sorted(declared))

    def test_every_script_with_marked_texts_in_the_web_folder_is_declared_by_exactly_one_page(self):
        from web_ui.Page_Texts import PAGE_CONTEXTS, read_page

        owners = collections.Counter()
        for descriptor in self.descriptors:
            for page in descriptor.pages:
                owners.update(Plugin_Manager.i18n_files(page, descriptor.src_root))
        self.assertEqual(set(owners), set(PAGE_CONTEXTS))
        self.assertEqual({name: count for name, count in owners.items() if count != 1}, {})
        for script in sorted(WEB_DIR.glob("*.js")):
            relative = f"web_ui/{script.name}"
            with self.subTest(script=relative):
                has_texts = bool(read_page(script.read_text(encoding="utf-8"), ".js").texts)
                self.assertEqual(relative in PAGE_CONTEXTS, has_texts, "a script with marked texts must be declared, and "
                                 "a declared one must have some")

    def test_every_local_script_a_page_loads_is_found_by_its_plugins_routes_and_declared_if_it_has_texts(self):
        from web_ui import agent_backend, esmfold_backend, meta_backend
        from web_ui.Page_Texts import PAGE_CONTEXTS, read_page

        registry = WebPluginRegistry(FakeViewer())
        with tempfile.TemporaryDirectory() as structures, \
                mock.patch.object(esmfold_backend, "get_structures_directory", return_value=structures):
            for plugin_id, backend in (("agent", agent_backend), ("esmfold", esmfold_backend), ("meta", meta_backend)):
                registry._active_registration_owner = plugin_id
                backend.register_backend(registry, FakeViewer())
        registry._active_registration_owner = None
        loaded, vendored_seen = 0, set()
        for page in self.pages:
            source = (WEB_DIR / page.path.lstrip("/")).read_text(encoding="utf-8")
            for address in self.SCRIPT.findall(source):
                if "://" in address:
                    continue
                path = address.split("?")[0]
                path = path if path.startswith("/") else "/" + path
                route = next(((prefix, folder) for prefix, folder in registry.static_routes.items()
                              if path.startswith(prefix)), None)
                file = Path(route[1]) / path[len(route[0]):] if route else WEB_DIR / path.lstrip("/")
                with self.subTest(page=page.path, script=address):
                    self.assertTrue(file.is_file(), f"{address} is not served from any folder")
                    loaded += 1
                    relative = file.resolve().relative_to(SRC_DIR.resolve()).as_posix()
                    if relative in self.VENDORED:
                        vendored_seen.add(relative)
                        self.assertNotIn(relative, PAGE_CONTEXTS)
                        continue
                    texts = read_page(file.read_text(encoding="utf-8"), ".js").texts
                    # A script with marked texts is one of its page's files; one without is declared by no page.
                    self.assertEqual(relative in PAGE_CONTEXTS, bool(texts), f"{relative}: a script that marks texts "
                                     "must be among the i18n files of its page, and one that does not must not be")
                    if texts:
                        self.assertEqual(PAGE_CONTEXTS[relative], page.i18n_context)
        self.assertGreater(loaded, 15)
        self.assertEqual(vendored_seen, set(self.VENDORED), "an entry of VENDORED that no page loads")

    def test_the_translation_update_stages_the_declared_files_and_nothing_else(self):
        from web_ui.Page_Texts import PAGE_CONTEXTS

        sys.path.insert(0, str(SRC_DIR / "resources" / "languages"))
        try:
            import Update_Translations
        finally:
            sys.path.remove(str(SRC_DIR / "resources" / "languages"))
        with tempfile.TemporaryDirectory() as stage:
            self.assertEqual(Update_Translations.stage_pages(SRC_DIR, stage), [])
            staged = {path.relative_to(stage).as_posix() for path in Path(stage).rglob("*") if path.is_file()}
            manifests = {f"web_ui/plugins/{path.name}" for path in BUNDLED_PLUGIN_DIR.glob("*.py") if path.name != "__init__.py"}
        self.assertEqual(staged - manifests, set(PAGE_CONTEXTS))
        self.assertEqual(len(staged - manifests), 15)
        self.assertEqual(staged & manifests, manifests)  # each manifest, for its pages' labels

    def test_the_labels_the_update_lists_are_the_ones_the_pages_are_opened_with(self):
        # The catalog lists each label as a Message; the opener shows it as one.
        sys.path.insert(0, str(SRC_DIR / "resources" / "languages"))
        try:
            import Update_Translations
        finally:
            sys.path.remove(str(SRC_DIR / "resources" / "languages"))
        from utilities.Localization import MESSAGE_CONTEXT, read_catalog

        listed = {message.source for message in read_catalog(Update_Translations.LANGUAGES_DIR / "emapssn.ts")
                  if message.context == MESSAGE_CONTEXT}
        for page in self.pages:
            with self.subTest(label=page.label):
                self.assertIn(page.label, listed)


class InitialWebStateTests(unittest.TestCase):
    """The state a page is sent when it connects: the Viewer's network and metadata, and each plugin's own keys."""

    BASE_KEYS = ["rows", "columns", "selected_indices", "visible_mask"]
    MODEL_KEYS = ["llm_loaded", "llm_backend", "llm_model_name"]

    @staticmethod
    def viewer(**attributes):
        import numpy as np

        viewer = SimpleNamespace(
            metadata={"Org": {"type": "text", "values": np.array(["x", "y"], dtype=object)}},
            selected_indices=[1], visible_mask=np.array([True, False]),
            get_serializable_metadata=lambda: [{"id": 0}, {"id": 1}], **attributes)
        return viewer

    def test_the_viewers_own_state_names_no_model(self):
        from EMAPSSN_Viewer import MainViewer

        viewer = self.viewer(llm_loaded=True, llm_backend="server", llm_model_name="x", web_plugin_registry=None)
        state = MainViewer.get_initial_web_state(viewer)
        self.assertEqual(list(state), self.BASE_KEYS)

    def test_the_agent_plugins_provider_adds_the_model_keys_straight_after_the_viewers(self):
        # The page is sent the same state, key for key, as when the Viewer held these three keys itself.
        from EMAPSSN_Viewer import MainViewer
        from web_ui import agent_backend, meta_backend

        for attributes, expected in (
            ({"llm_loaded": True, "llm_backend": "server", "llm_model_name": "fake-model"}, [True, "server", "fake-model"]),
            ({}, [False, None, "Unknown"]),
        ):
            with self.subTest(attributes=attributes):
                viewer = self.viewer(**attributes)
                registry = viewer.web_plugin_registry = WebPluginRegistry(viewer)
                agent_backend.register_backend(registry, viewer)
                meta_backend.register_backend(registry, viewer)
                with mock.patch.object(agent_backend, "load_agent_history", return_value=[{"role": "user"}]):
                    state = MainViewer.get_initial_web_state(viewer)
                self.assertEqual(list(state)[:7], self.BASE_KEYS + self.MODEL_KEYS)
                self.assertEqual([state[key] for key in self.MODEL_KEYS], expected)
                self.assertEqual(state["llm_history"], [{"role": "user"}])
                self.assertEqual(state["types"], {"Org": "text"})

    def test_the_models_keys_are_the_ones_the_page_is_told_when_the_model_changes(self):
        from web_ui import agent_backend

        viewer = SimpleNamespace(llm_loaded=True, llm_backend="server", llm_model_name="m", broadcast_event=mock.Mock())
        agent_backend.broadcast_backend_state(viewer)
        event = viewer.broadcast_event.call_args.args[0]
        self.assertEqual(list(event)[:4], ["type", *self.MODEL_KEYS])
        self.assertEqual({key: event[key] for key in self.MODEL_KEYS}, {"llm_loaded": True, "llm_backend": "server", "llm_model_name": "m"})


if __name__ == "__main__":
    unittest.main()
