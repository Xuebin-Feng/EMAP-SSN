"""A real hidden Viewer, started and driven through the MCP client (mode="hidden").

Launches the Viewer on the two-node fixture of tests/test_mcp_viewer_lifecycle.py on
Qt's native platform with its window kept off screen, and checks what hidden mode is
for: capture_view and print render while the window is hidden, the window comes on
screen and goes off again with its state intact, and closing a shown window ends
the Viewer. They need a display and an OpenGL context: they skip when the host has
no display, and when the hidden Viewer falls back to headless because the native
platform could not make a context. tests/test_viewer_hidden_window.py has the unit tests.
"""
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from PIL import Image
from tests import test_mcp_viewer_lifecycle as lifecycle
from tests import layout_fixtures as fixtures
from mcp_server.viewer.Viewer_Client import MCPViewerClient
from utilities.Viewer_Display import display_unavailable_reason


def visible_windows(pids):
    """Titles of the visible top-level Windows windows of ``pids`` (Windows only)."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    titles = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def collect(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value in pids and user32.IsWindowVisible(hwnd):
            title = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, title, 512)
            if title.value:
                titles.append((hwnd, title.value))
        return True

    user32.EnumWindows(collect, 0)
    return titles


def process_tree(pid):
    try:
        root = psutil.Process(pid)
        return {process.pid for process in [root, *root.children(recursive=True)]}
    except psutil.Error:
        return set()


class HiddenViewerTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self):
        fixture = lifecycle.SettingsTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        sessions = fixture.root / "sessions"
        self.addCleanup(lifecycle._stop_recorded_launches, sessions)
        self.results = fixture.root / "results"
        fixture.document["directories"]["ANALYSIS_RESULT_DIR"] = str(self.results)
        return fixture, sessions

    async def capture(self, client, session_id):
        reply = await client.command_action("capture_view", {}, session_id=session_id)
        image = Image.open(io.BytesIO(base64.b64decode(reply["image_base64"]))).convert("RGB")
        self.assertGreater(image.width, 100)
        self.assertGreater(image.height, 100)
        # A canvas that rendered holds nodes, text and the slider: many colours, not a blank frame.
        self.assertGreater(len(image.getcolors(maxcolors=1_000_000) or []), 20)
        return image

    async def run_command(self, client, session_id, text, name):
        record = await client.command_action("execute_commands", {"submission_id": name, "commands": [text]},
                                             session_id=session_id)
        for _ in range(300):
            if record["status"] in {"succeeded", "failed", "cancelled", "skipped"}:
                break
            await asyncio.sleep(0.2)
            record = await client.command_action("get_command_request", {"request_id": record["request_id"]},
                                                 session_id=session_id)
        return record

    async def test_a_hidden_viewer_renders_hides_shows_and_closes_with_its_window(self):
        reason = display_unavailable_reason()
        if reason:
            self.skipTest(f"A hidden Viewer needs a display: {reason}.")
        fixture, sessions = self.fixture()
        with mock.patch.dict(os.environ, lifecycle._launch_environment(SSN_VIEWER_SESSION_DIR=str(sessions)),
                             clear=True):
            client = MCPViewerClient(fixtures.ROOT)
            started = time.monotonic()
            info = await client.launch_session(settings_document=fixture.document, mode="hidden", timeout=120)
            self.assertEqual(info["status"], "ready", info)
            session_id, pid = info["session_id"], info["pid"]
            if info.get("fallback_reason"):
                await client.close_session(session_id)
                self.skipTest(f"No OpenGL context on this host's native platform: {info['fallback_reason']}")
            print(f"hidden Viewer ready in {time.monotonic() - started:.1f} s, "
                  f"{sum(process.memory_info().rss for process in map(psutil.Process, process_tree(pid))) / 2**20:.0f} MB RSS",
                  file=sys.stderr)
            self.assertEqual(info["mode"], "hidden")
            self.assertNotIn("requested_mode", info)
            on_windows = sys.platform == "win32"
            window = (await client.get_summary(session_id))["window"]
            self.assertEqual((window["mode"], window["visible"], window["can_show"], window["can_hide"]),
                             ("hidden", False, True, True))
            if on_windows:
                self.assertEqual(visible_windows(process_tree(pid)), [], "a hidden Viewer opens no window")

            # Rendering works with the window hidden: capture_view and print.
            hidden_image = await self.capture(client, session_id)
            printed = await self.run_command(client, session_id, "print hidden.png", "print-hidden")
            self.assertEqual(printed["status"], "succeeded", printed)
            picture = self.results / "Saved_Images" / "hidden.png"
            for _ in range(150):
                if picture.exists() and picture.stat().st_size:
                    break
                await asyncio.sleep(0.2)
            self.assertTrue(picture.read_bytes().startswith(b"\x89PNG"), "print wrote no PNG while hidden")

            # A dialog needs the user: refused while the window is hidden.
            refused = await self.run_command(client, session_id, "alignment", "dialog-hidden")
            self.assertEqual(refused["status"], "failed", refused)
            self.assertIn("Show the window first", json.dumps(refused))

            shown = (await client.window_action("show", session_id))["window"]
            self.assertTrue(shown["visible"], shown)
            self.assertFalse(shown["minimized"])
            if on_windows:
                titles = [title for _, title in visible_windows(process_tree(pid))]
                self.assertEqual(len(titles), 1, titles)
                self.assertIn(f"[{info['session_alias']}]", titles[0])
            await self.capture(client, session_id)
            self.assertTrue((await client.get_summary(session_id))["window"]["visible"])

            hidden = (await client.window_action("hide", session_id))["window"]
            self.assertFalse(hidden["visible"], hidden)
            if on_windows:
                self.assertEqual(visible_windows(process_tree(pid)), [])
            await self.capture(client, session_id)
            self.assertEqual((await client.get_summary(session_id))["node_count"], 2)
            self.assertEqual(hidden_image.size, (await self.capture(client, session_id)).size)

            # Closing a window the client showed ends the Viewer, as for any Viewer.
            self.assertTrue((await client.window_action("show", session_id))["window"]["visible"])
            if on_windows:
                import ctypes
                from ctypes import wintypes
                user32 = ctypes.windll.user32
                user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
                handle = visible_windows(process_tree(pid))[0][0]
                self.assertTrue(user32.PostMessageW(handle, 0x0010, 0, 0))   # WM_CLOSE
                deadline = time.monotonic() + 15
                while psutil.pid_exists(pid) and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                self.assertFalse(psutil.pid_exists(pid), "closing the shown window did not end the Viewer")
            else:
                await client.close_session(session_id)
                deadline = time.monotonic() + 15
                while psutil.pid_exists(pid) and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                self.assertFalse(psutil.pid_exists(pid))

    @unittest.skipUnless(sys.platform.startswith("linux"), "the no-display case is checked for real on Linux")
    async def test_a_hidden_viewer_without_a_display_runs_headless_and_says_why(self):
        fixture, sessions = self.fixture()
        environment = lifecycle._launch_environment(SSN_VIEWER_SESSION_DIR=str(sessions))
        for name in ("DISPLAY", "WAYLAND_DISPLAY"):
            environment.pop(name, None)
        with mock.patch.dict(os.environ, environment, clear=True):
            client = MCPViewerClient(fixtures.ROOT)
            info = await client.launch_session(settings_document=fixture.document, mode="hidden", timeout=120)
            self.assertEqual(info["status"], "ready", info)
            session_id = info["session_id"]
            try:
                self.assertEqual((info["mode"], info["requested_mode"]), ("headless", "hidden"))
                self.assertIn("no display was found", info["fallback_reason"])
                window = (await client.get_summary(session_id))["window"]
                self.assertEqual((window["mode"], window["requested_mode"], window["can_show"]), ("headless", "hidden", False))
                for action in ("show", "hide"):
                    with self.assertRaisesRegex(Exception, "no display was found"):
                        await client.window_action(action, session_id)
                # Everything but rendering works, and rendering says what is missing.
                with self.assertRaisesRegex(Exception, "no display was found"):
                    await client.command_action("capture_view", {}, session_id=session_id)
                printed = await self.run_command(client, session_id, "print nodisplay.png", "print-nodisplay")
                self.assertEqual(printed["status"], "failed", printed)
                self.assertIn("no display was found", json.dumps(printed))
                self.assertEqual((await client.get_summary(session_id))["node_count"], 2)
            finally:
                await client.close_session(session_id)


if __name__ == "__main__":
    unittest.main()
