import importlib.util
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_PATH = os.path.join(PROJECT_ROOT, "src", "web_ui", "esmfold_backend.py")

config_stub = types.ModuleType("SSN_Config")
config_stub.resolve_directory_path = lambda value: value
plugin_manager_stub = types.ModuleType("web_ui.Plugin_Manager")
plugin_manager_stub.ensure_registry = mock.Mock()
cache_selection_stub = types.ModuleType("utilities.Cache_Selection")
cache_selection_stub.resolve_selected_cache = mock.Mock()

spec = importlib.util.spec_from_file_location("esmfold_backend_under_test", BACKEND_PATH)
esmfold_backend = importlib.util.module_from_spec(spec)
with mock.patch.dict(
    sys.modules,
    {
        "SSN_Config": config_stub,
        "web_ui.Plugin_Manager": plugin_manager_stub,
        "utilities.Cache_Selection": cache_selection_stub,
    },
):
    spec.loader.exec_module(esmfold_backend)


class FakeWebServer:
    def __init__(self, connected_clients=()):
        self.connected_clients = set(connected_clients)

    def has_event_client(self, client_id):
        return client_id in self.connected_clients


class ESMFoldBrowserOpeningTests(unittest.TestCase):
    @staticmethod
    def make_viewer(connected_clients=()):
        return SimpleNamespace(
            console_text=SimpleNamespace(text=""),
            web_server=FakeWebServer(connected_clients),
            get_web_url=lambda path: f"http://localhost:49123/{path.lstrip('/')}",
        )

    def test_fallback_routes_esmfold_through_shared_opener(self):
        viewer = self.make_viewer()
        with mock.patch.object(
            esmfold_backend, "open_browser_page", return_value=True
        ) as shared_open:
            self.assertTrue(esmfold_backend.open_esmfold_ui(viewer))

        shared_open.assert_called_once_with(
            viewer,
            "/esmfold.html",
            "ESMFold Mol* UI",
            "esmfold",
            show_existing_dialog=True,
        )

    def test_viewer_shared_opener_is_preferred(self):
        viewer = self.make_viewer()
        viewer._open_web_ui = mock.Mock(return_value=False)
        with mock.patch.object(esmfold_backend, "open_browser_page") as fallback:
            self.assertFalse(esmfold_backend.open_esmfold_ui(viewer))

        viewer._open_web_ui.assert_called_once_with(
            "/esmfold.html",
            "ESMFold Mol* UI",
            "esmfold",
            show_existing_dialog=True,
        )
        fallback.assert_not_called()

    def test_non_modal_request_is_propagated_to_viewer_opener(self):
        viewer = self.make_viewer()
        viewer._open_web_ui = mock.Mock(return_value=False)

        self.assertFalse(
            esmfold_backend.open_esmfold_ui(
                viewer,
                show_existing_dialog=False,
            )
        )

        viewer._open_web_ui.assert_called_once_with(
            "/esmfold.html",
            "ESMFold Mol* UI",
            "esmfold",
            show_existing_dialog=False,
        )

    def test_fold_view_sidebar_uses_default_modal_behavior(self):
        viewer = SimpleNamespace(add_sidebar_button=mock.Mock())
        esmfold_backend.activate(viewer)
        callback = viewer.add_sidebar_button.call_args.args[2]

        with mock.patch.object(esmfold_backend, "open_esmfold_ui") as open_ui:
            callback()

        open_ui.assert_called_once_with(viewer)


if __name__ == "__main__":
    unittest.main()
