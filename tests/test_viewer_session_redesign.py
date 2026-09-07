import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import test_layout_cache_generator as fixtures
from utilities.Viewer_Settings import validate_viewer_document, ViewerSettingsError, normalize_viewer_settings
from mcp_server.MCP_Viewer_Client import MCPViewerClient, MCPViewerError
from EMAPSSN_MCP_Server import mcp
from mcp import Client, StdioServerParameters


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixtures._write_inputs(self.root)
        settings = fixtures.LayoutGenerationSettings.from_document(fixtures._settings_document(self.root), project_root=fixtures.ROOT)
        with mock.patch("Layout_Engine_SSN.calculate_layout", return_value=(np.array([[0, 0], [1, 1]], dtype=np.float32), 10.0)):
            self.cache = fixtures.generate_layout_cache(settings).cache_path
        self.document = dict(TARGET_CACHE_PATH=self.cache, NODE_FASTA_FILE=str(self.root / "set.fasta"),
            INPUT_HDF5=str(self.root / "network.h5"), MSA_FILE="", UMAP_MODE=False,
            ALIGNMENT_SCORE="global", NORM_MODE="alignment_length", SIMILARITY_THRESHOLD=0.1,
            TOP_EDGE_PERCENT=None)

    def test_complete_document_and_no_personal_settings(self):
        with mock.patch.dict(os.environ, {"SSN_VIEWER_SETTINGS_PATH": "unrelated.json"}):
            result = validate_viewer_document(self.document, fixtures.ROOT)
        self.assertEqual(result["MSA_FILE"], "")
        self.assertEqual(result["NODE_SIZE"], 10)
        self.assertEqual(result["SIMILARITY_THRESHOLD"], 0.1)

    def test_missing_files_fields_and_manifest_conflicts(self):
        for key in ("NODE_FASTA_FILE", "INPUT_HDF5", "MSA_FILE", "UMAP_MODE", "ALIGNMENT_SCORE"):
            doc = dict(self.document); doc.pop(key)
            with self.subTest(key=key), self.assertRaises(ViewerSettingsError):
                validate_viewer_document(doc, fixtures.ROOT)
        for override in ({"INPUT_HDF5": "missing.h5"}, {"SIMILARITY_THRESHOLD": 100},
                         {"ALIGNMENT_REFERENCE": "missing"}, {"NODE_SIZE": True}, {"TYPO": 1}):
            with self.subTest(override=override), self.assertRaises(ViewerSettingsError):
                validate_viewer_document(dict(self.document, **override), fixtures.ROOT)

    def test_aliases_and_header_order(self):
        doc = dict(self.document, INPUT_FILE_DIR=str(self.root), NODE_FASTA_FILE="$input_file$/set.fasta",
                   INPUT_HDF5="$input_file$/network.h5")
        self.assertEqual(validate_viewer_document(doc, fixtures.ROOT)["INPUT_HDF5"], str(self.root / "network.h5"))
        import h5py
        with h5py.File(self.cache, "r+") as cache:
            cache["headers"][:] = cache["headers"][:][::-1]
        with self.assertRaises(ViewerSettingsError):
            validate_viewer_document(doc, fixtures.ROOT)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_visible_viewer_launch_through_headless_config(self):
        fixture = SettingsTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": str(fixture.root / "sessions"),
                                         "PYTHONIOENCODING": "utf-8"}):
            client = MCPViewerClient(fixtures.ROOT)
            info = None
            try:
                info = await client.launch_session(settings_document=fixture.document, mode="normal", timeout=45)
                self.assertEqual(info["mode"], "normal")
                summary = await client.get_summary(info["session_id"])
                self.assertEqual(summary["node_count"], 2)
                output = await client.read_log(info["session_id"])
                self.assertTrue(output["text"], "Visible Viewer must retain startup terminal output")
                self.assertIn(f'[{info["session_alias"]}]', summary["window_title"])
                self.assertEqual(summary["cache_metadata"]["status"], "complete")
                connected = await client.connect_session(info["session_alias"].lower())
                self.assertEqual(connected["session_id"], info["session_id"])
                listed = await client.list_sessions()
                item = next(s for s in listed["sessions"] if s["session_id"] == info["session_id"])
                self.assertEqual(item["cache_metadata"]["cache_path"], fixture.cache)
                self.assertEqual(item["session_alias"], info["session_alias"])
                if sys.platform == "win32":
                    import ctypes
                    import subprocess
                    from ctypes import wintypes
                    identity = json.loads((Path(info["stdout_log"]).parent / "terminal-process.json").read_text())
                    console = subprocess.run([sys.executable, "-c",
                        "import ctypes,sys; k=ctypes.windll.kernel32; k.FreeConsole(); "
                        "attached=k.AttachConsole(int(sys.argv[1])); k.GetConsoleWindow.restype=ctypes.c_void_p; "
                        "h=k.GetConsoleWindow(); print(bool(attached and h)); k.FreeConsole()",
                        str(identity["pid"])], capture_output=True, text=True, timeout=5)
                    self.assertEqual(console.stdout.strip(), "True", console.stderr)
                    user32 = ctypes.windll.user32
                    windows = []
                    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
                    @callback_type
                    def collect(hwnd, _):
                        pid = wintypes.DWORD()
                        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                        title = ctypes.create_unicode_buffer(512)
                        user32.GetWindowTextW(hwnd, title, 512)
                        if pid.value == info["pid"] and info["session_alias"] in title.value:
                            windows.append(hwnd)
                        return True
                    user32.EnumWindows(collect, 0)
                    self.assertTrue(windows, "Expected a native Viewer window")
                    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
                    self.assertTrue(user32.PostMessageW(windows[0], 0x0010, 0, 0))  # WM_CLOSE, as with X
                    deadline = asyncio.get_running_loop().time() + 10
                    while client.connected_session_id and asyncio.get_running_loop().time() < deadline:
                        await asyncio.sleep(0.1)
                    self.assertIsNone(client.connected_session_id)
                    self.assertTrue((await client.read_log(info["session_id"]))["text"])
                    self.assertFalse(psutil.pid_exists(info["pid"]))
                    info = None
            finally:
                if info:
                    await client.close_session(info["session_id"])
                    self.assertFalse(psutil.pid_exists(info["pid"]))

    async def test_timeout_and_cancellation_cleanup_verbose_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src" / "EMAPSSN_Config.py").write_text(
                "import time,sys\nprint('x'*1000000,flush=True)\ntime.sleep(60)\n"
            )
            with mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": str(root / "sessions")}), mock.patch(
                "mcp_server.MCP_Viewer_Client.validate_viewer_document", return_value={"TARGET_CACHE_PATH": "test"}
            ):
                client = MCPViewerClient(root)
                with self.assertRaisesRegex(MCPViewerError, "Timed out"):
                    await client.launch_session(settings_document={}, mode="headless", timeout=0.3)
                task = asyncio.create_task(client.launch_session(settings_document={}, mode="headless"))
                deadline = asyncio.get_running_loop().time() + 10
                while not any(p.stat().st_size >= 1000000 for p in (root / "sessions").rglob("stdout.log")):
                    if task.done() or asyncio.get_running_loop().time() >= deadline:
                        break
                    await asyncio.sleep(0.1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            for path in (root / "sessions").rglob("process.json"):
                identity = json.loads(path.read_text())
                self.assertFalse(psutil.pid_exists(identity["pid"]))
            self.assertTrue(any(p.stat().st_size >= 1000000 for p in (root / "sessions").rglob("stdout.log")))

    async def test_stdio_survival_connection_isolation_and_verified_close(self):
        fixture = SettingsTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        directory = fixture.root / "sessions"
        environment = dict(os.environ, SSN_VIEWER_SESSION_DIR=str(directory), PYTHONIOENCODING="utf-8",
                           TEMP=str(fixture.root), TMP=str(fixture.root))
        parameters = StdioServerParameters(command=sys.executable,
            args=["-u", str(fixtures.SRC / "EMAPSSN_MCP_Server.py")], cwd=str(fixtures.ROOT), env=environment)
        info = None
        with mock.patch.dict(os.environ, environment), mock.patch.object(tempfile, "tempdir", str(fixture.root)), mock.patch("mcp.os.win32.utilities._create_job_object", return_value=None):
            try:
                async with Client(parameters, read_timeout_seconds=60) as first:
                    started = await first.call_tool("start_viewer_session", {"mode": "headless", "settings_document": fixture.document})
                    self.assertFalse(started.is_error, str(started))
                    info = started.structured_content
                    output = await first.call_tool("read_viewer_log")
                    self.assertFalse(output.is_error, str(output))
                    self.assertTrue(output.structured_content["text"])
                    async with Client(mcp) as second:
                        self.assertTrue((await second.call_tool("get_viewer_summary")).is_error)
                        connected = await second.call_tool("connect_viewer_session", {"session_id": info["session_id"]})
                        self.assertFalse(connected.is_error, str(connected))
                        self.assertFalse((await second.call_tool("disconnect_viewer_session")).is_error)
                        self.assertTrue((await second.call_tool("get_viewer_summary")).is_error)
                        response = await first.call_tool("get_viewer_summary")
                        self.assertFalse(response.is_error, str(response))
                # The STDIO server is gone; the independent Viewer must still answer.
                async with Client(parameters, read_timeout_seconds=60) as third:
                    connected = await third.call_tool("connect_viewer_session", {"session_id": info["session_id"]})
                    self.assertFalse(connected.is_error, str(connected))
                    summary = await third.call_tool("get_viewer_summary")
                    self.assertEqual(summary.structured_content["node_count"], 2)
                    closed = await third.call_tool("close_viewer_session")
                    self.assertFalse(closed.is_error, str(closed))
                    self.assertFalse(psutil.pid_exists(info["pid"]))
                    info = None
            finally:
                if info is not None:
                    if psutil.pid_exists(info["pid"]):
                        try:
                            await MCPViewerClient().close_session(info["session_id"])
                        except MCPViewerError:
                            try:
                                process = psutil.Process(info["pid"]); process.terminate(); process.wait(5)
                            except psutil.NoSuchProcess:
                                pass

    async def test_disconnect_and_failed_close_preserve_state(self):
        client = MCPViewerClient()
        client.connected_session_id = "existing"
        with mock.patch("mcp_server.MCP_Viewer_Client.select_viewer_session", side_effect=LookupError("unreachable")):
            with self.assertRaises(MCPViewerError):
                await client.close_session()
        self.assertEqual(client.connected_session_id, "existing")
        self.assertEqual((await client.disconnect_session())["session_id"], "existing")
        self.assertIsNone((await client.disconnect_session())["session_id"])
        with self.assertRaises(MCPViewerError):
            await client.get_summary()

    async def test_failed_termination_does_not_remove_descriptor(self):
        from types import SimpleNamespace
        session = SimpleNamespace(session_id="test", pid=123, process_created_at=None,
                                  base_url="http://127.0.0.1:9", token="test")
        client = MCPViewerClient()
        client.connected_session_id = "test"
        with mock.patch("mcp_server.MCP_Viewer_Client.select_viewer_session", return_value=session), mock.patch(
            "mcp_server.MCP_Viewer_Client._identity", return_value=mock.Mock()
        ), mock.patch("mcp_server.MCP_Viewer_Client._alive", return_value=True), mock.patch(
            "mcp_server.MCP_Viewer_Client.urllib.request.urlopen", side_effect=OSError("offline")
        ), mock.patch("mcp_server.MCP_Viewer_Client._terminate_tree", side_effect=psutil.AccessDenied(123)), mock.patch(
            "mcp_server.MCP_Viewer_Client.remove_viewer_session"
        ) as remove:
            with self.assertRaises(MCPViewerError):
                await client.close_session(timeout=0)
            remove.assert_not_called()
        self.assertEqual(client.connected_session_id, "test")


if __name__ == "__main__":
    unittest.main()
