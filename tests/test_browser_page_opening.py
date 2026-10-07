"""Opening the bundled browser pages: one tab per page and Viewer
(src/web_ui/Browser_Page.py), each Viewer's own URL, and the ESMFold page's
opener (src/web_ui/esmfold_backend.py)."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from commands import agent, meta  # noqa: E402
from EMAPSSN_Viewer import MainViewer  # noqa: E402
from web_ui import Browser_Page, esmfold_backend  # noqa: E402


class FakeWebServer:
    def __init__(self, connected_clients=()):
        self.connected_clients = set(connected_clients)

    def has_event_client(self, client_id):
        return client_id in self.connected_clients


class BrowserPageOpeningTests(unittest.TestCase):
    @staticmethod
    def make_viewer(connected_clients=()):
        return SimpleNamespace(
            console_text=SimpleNamespace(text=""),
            main_window=object(),
            update_console_background=mock.Mock(),
            web_server=FakeWebServer(connected_clients),
            get_web_url=lambda path: f"http://localhost:49123/{path.lstrip('/')}",
        )

    def test_disconnected_client_opens_one_new_tab(self):
        viewer = self.make_viewer()
        with mock.patch.object(
            Browser_Page.webbrowser, "open", return_value=True
        ) as browser_open:
            self.assertTrue(
                Browser_Page.open_browser_page(
                    viewer, "/meta.html", "Metadata UI", "meta"
                )
            )

        browser_open.assert_called_once_with(
            "http://localhost:49123/meta.html", new=2
        )
        self.assertEqual(
            viewer.console_text.text,
            "Metadata UI opened at http://localhost:49123/meta.html",
        )
        self.assertIn("meta", viewer._web_ui_pending_opens)

    def test_matching_connected_client_reports_without_opening(self):
        viewer = self.make_viewer({"meta"})
        with (
            mock.patch.object(Browser_Page.webbrowser, "open") as browser_open,
            mock.patch.object(
                Browser_Page.QMessageBox, "information"
            ) as information,
        ):
            self.assertFalse(
                Browser_Page.open_browser_page(
                    viewer, "/meta.html", "Metadata UI", "meta"
                )
            )

        browser_open.assert_not_called()
        message = "Metadata UI is already open in your browser."
        self.assertEqual(viewer.console_text.text, message)
        viewer.update_console_background.assert_called_once_with()
        information.assert_called_once_with(
            viewer.main_window,
            "Browser Page Already Open",
            message,
        )

    def test_connected_client_can_report_without_dialog(self):
        viewer = self.make_viewer({"esmfold"})
        with (
            mock.patch.object(Browser_Page.webbrowser, "open") as browser_open,
            mock.patch.object(
                Browser_Page.QMessageBox, "information"
            ) as information,
        ):
            self.assertFalse(
                Browser_Page.open_browser_page(
                    viewer,
                    "/esmfold.html",
                    "ESMFold Mol* UI",
                    "esmfold",
                    show_existing_dialog=False,
                )
            )

        browser_open.assert_not_called()
        information.assert_not_called()
        self.assertEqual(
            viewer.console_text.text,
            "ESMFold Mol* UI is already open in your browser.",
        )

    def test_unrelated_connected_client_does_not_suppress_opening(self):
        viewer = self.make_viewer({"agent"})
        with mock.patch.object(
            Browser_Page.webbrowser, "open", return_value=True
        ) as browser_open:
            self.assertTrue(
                Browser_Page.open_browser_page(
                    viewer, "/meta.html", "Metadata UI", "meta"
                )
            )

        browser_open.assert_called_once()

    def test_pending_open_blocks_rapid_second_click(self):
        viewer = self.make_viewer()
        with (
            mock.patch.object(
                Browser_Page.webbrowser, "open", return_value=True
            ) as browser_open,
            mock.patch.object(
                Browser_Page.QMessageBox, "information"
            ) as information,
            mock.patch.object(Browser_Page.time, "monotonic", return_value=100.0),
        ):
            self.assertTrue(
                Browser_Page.open_browser_page(
                    viewer, "/agent.html", "Agent UI", "agent"
                )
            )
            self.assertFalse(
                Browser_Page.open_browser_page(
                    viewer, "/agent.html", "Agent UI", "agent"
                )
            )

        browser_open.assert_called_once()
        message = "Agent UI is already being opened in your browser."
        self.assertEqual(viewer.console_text.text, message)
        information.assert_called_once_with(
            viewer.main_window,
            "Browser Page Already Open",
            message,
        )

    def test_pending_open_can_report_without_dialog(self):
        viewer = self.make_viewer()
        viewer._web_ui_pending_opens = {"esmfold": 110.0}
        with (
            mock.patch.object(Browser_Page.time, "monotonic", return_value=100.0),
            mock.patch.object(Browser_Page.webbrowser, "open") as browser_open,
            mock.patch.object(
                Browser_Page.QMessageBox, "information"
            ) as information,
        ):
            self.assertFalse(
                Browser_Page.open_browser_page(
                    viewer,
                    "/esmfold.html",
                    "ESMFold Mol* UI",
                    "esmfold",
                    show_existing_dialog=False,
                )
            )

        browser_open.assert_not_called()
        information.assert_not_called()
        self.assertEqual(
            viewer.console_text.text,
            "ESMFold Mol* UI is already being opened in your browser.",
        )

    def test_expired_pending_open_allows_retry(self):
        viewer = self.make_viewer()
        viewer._web_ui_pending_opens = {"agent": 100.0}
        with (
            mock.patch.object(Browser_Page.time, "monotonic", return_value=100.1),
            mock.patch.object(
                Browser_Page.webbrowser, "open", return_value=True
            ) as browser_open,
        ):
            self.assertTrue(
                Browser_Page.open_browser_page(
                    viewer, "/agent.html", "Agent UI", "agent"
                )
            )

        browser_open.assert_called_once()

    def test_closed_sse_connection_allows_reopening(self):
        viewer = self.make_viewer({"esmfold"})
        with (
            mock.patch.object(
                Browser_Page.QMessageBox, "information"
            ),
            mock.patch.object(
                Browser_Page.webbrowser, "open", return_value=True
            ) as browser_open,
        ):
            self.assertFalse(
                Browser_Page.open_browser_page(
                    viewer,
                    "/esmfold.html",
                    "ESMFold Mol* UI",
                    "esmfold",
                )
            )
            viewer.web_server.connected_clients.remove("esmfold")
            self.assertTrue(
                Browser_Page.open_browser_page(
                    viewer,
                    "/esmfold.html",
                    "ESMFold Mol* UI",
                    "esmfold",
                )
            )

        browser_open.assert_called_once()

    def test_failed_browser_launch_clears_pending_reservation(self):
        viewer = self.make_viewer()
        with mock.patch.object(
            Browser_Page.webbrowser, "open", return_value=False
        ):
            self.assertFalse(
                Browser_Page.open_browser_page(
                    viewer, "/meta.html", "Metadata UI", "meta"
                )
            )

        self.assertNotIn("meta", viewer._web_ui_pending_opens)
        self.assertEqual(
            viewer.console_text.text,
            "Could not open Metadata UI: http://localhost:49123/meta.html",
        )


class InstanceUrlRoutingTests(unittest.TestCase):
    class Viewer:
        # MainViewer's own openers, so the URLs opened are the ones production builds.
        _open_web_ui = MainViewer._open_web_ui
        open_agent_ui = MainViewer.open_agent_ui
        open_metadata_ui = MainViewer.open_metadata_ui

        def __init__(self):
            self.console_text = mock.Mock(text="")
            self.web_server = None

        def get_web_url(self, path):
            return f"http://127.0.0.1:49123/{path.lstrip('/')}"

        def update_console_background(self):
            pass

    def test_agent_meta_and_esmfold_use_viewer_instance_port(self):
        viewer = self.Viewer()
        expected = [
            "http://127.0.0.1:49123/agent.html",
            "http://127.0.0.1:49123/meta.html",
            "http://127.0.0.1:49123/esmfold.html",
        ]
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(agent, "register"), \
                mock.patch.object(meta, "register"), \
                mock.patch.object(meta.cfg, "METADATA_DIR", temp_dir), \
                mock.patch("webbrowser.open") as browser_open:
            agent.run(viewer, [])
            meta.run(viewer, [])
            esmfold_backend.open_esmfold_ui(viewer)

        self.assertEqual(
            [call.args[0] for call in browser_open.call_args_list], expected
        )


class ESMFoldBrowserOpeningTests(unittest.TestCase):
    def test_esmfold_page_opens_through_viewer_shared_opener(self):
        # The default suits the Fold View button; the esmfold command passes
        # show_existing_dialog=False.
        for options, show_existing_dialog in (({}, True), ({"show_existing_dialog": False}, False)):
            with self.subTest(options=options):
                viewer = SimpleNamespace(_open_web_ui=mock.Mock(return_value=False))
                self.assertFalse(esmfold_backend.open_esmfold_ui(viewer, **options))

                viewer._open_web_ui.assert_called_once_with(
                    "/esmfold.html",
                    "ESMFold Mol* UI",
                    "esmfold",
                    show_existing_dialog=show_existing_dialog,
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
