"""Focused tests for duplicate-safe bundled browser page opening."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from web_ui import Browser_Page


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


if __name__ == "__main__":
    unittest.main()
