import asyncio
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from mcp_server.viewer.Viewer_Client import MCPViewerClient, MCPViewerError


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_window_close_clears_selection_without_another_tool_call(self):
        client = MCPViewerClient()
        client.connected_session_id = "closed"
        with mock.patch("mcp_server.viewer.Viewer_Client.discover_viewer_sessions", return_value=[]):
            await asyncio.wait_for(client._monitor_connection(), 4)
        self.assertIsNone(client.connected_session_id)

    async def test_backend_transport_close_clears_selection(self):
        client = MCPViewerClient(transport_closed=lambda: True)
        client.connected_session_id = "still-running"
        await client._monitor_connection()
        self.assertIsNone(client.connected_session_id)

    async def test_list_and_query_clear_missing_selection(self):
        client = MCPViewerClient()
        client.connected_session_id = "closed"
        with mock.patch("mcp_server.viewer.Viewer_Client.discover_viewer_sessions", return_value=[]):
            self.assertIsNone((await client.list_sessions())["connected_session_id"])
        client.connected_session_id = "closed"
        with mock.patch("mcp_server.viewer.Viewer_Client.select_viewer_session", side_effect=LookupError("gone")):
            with self.assertRaises(MCPViewerError):
                await client.get_summary()
        self.assertIsNone(client.connected_session_id)

    async def test_retained_logs_after_disconnect_and_bounded_paging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stdout.log").write_bytes(b"first\nsecond\n")
            client = MCPViewerClient()
            client.connected_session_id = "test"
            client._log_directories["test"] = root
            await client.disconnect_session()
            first = await client.read_log("test", limit=6)
            second = await client.read_log("test", offset=first["next_offset"])
            self.assertEqual(first["text"], "first\n")
            self.assertEqual(second["text"], "second\n")
            self.assertTrue(second["eof"])
            with self.assertRaises(MCPViewerError):
                await client.read_log("test", stream="../settings")


class OutputTests(unittest.TestCase):
    def test_viewer_prints_any_text_whatever_the_pipe_encoding(self):
        """An MCP-launched Viewer prints to a pipe its launch monitor reads, which Python
        opens in the ANSI code page on Windows; a command echoing text outside
        it failed with UnicodeEncodeError."""
        from desktop import Desktop_Launcher_Monitor

        expression = "header == 'α-amylase 中文'"
        viewer = SRC / "EMAPSSN_Viewer.py"
        child = "\n".join([
            "import runpy, sys, types",
            f"sys.path.insert(0, {str(SRC)!r})",
            f"sys.argv = [{str(viewer)!r}, '--help']",
            "try:",
            "    runpy.run_path(sys.argv[0], run_name='__main__')  # the Viewer's startup",
            "except SystemExit:",
            "    pass",
            "import Command_Engine",
            "Command_Engine.report_selection_error(",
            f"    types.SimpleNamespace(), {ascii(expression)}, ValueError('No node matches.'))",
            "sys.exit(3)",  # not 0, so the monitor keeps its log and returns at once
        ])
        # cp1252 is that pipe encoding here; the variable sets it on any platform.
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONUTF8"}
        environment["PYTHONIOENCODING"] = "cp1252"
        start = subprocess.Popen
        with tempfile.TemporaryDirectory() as directory:
            launch = Path(directory) / "launch"
            environment[Desktop_Launcher_Monitor.SUPERVISED_DIRECTORY_VARIABLE] = str(launch)
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                Desktop_Launcher_Monitor.subprocess, "Popen",
                side_effect=lambda command, **kwargs: start([sys.executable, "-u", "-c", child], **kwargs),
            ):
                code = Desktop_Launcher_Monitor.run_monitor("session", Path(directory) / "state")
            log = (launch / "stdout.log").read_bytes()
            errors = (launch / "stderr.log").read_text(encoding="utf-8", errors="replace")
        self.assertEqual(code, 3, errors)
        self.assertIn(f"Expression: {expression}".encode("utf-8"), log)

if __name__ == "__main__":
    unittest.main()
