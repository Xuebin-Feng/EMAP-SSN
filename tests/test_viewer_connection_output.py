import asyncio
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from mcp_server.MCP_Viewer_Client import MCPViewerClient, MCPViewerError
from mcp_server.Viewer_Terminal import copy_output


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_window_close_clears_selection_without_another_tool_call(self):
        client = MCPViewerClient()
        client.connected_session_id = "closed"
        with mock.patch("mcp_server.MCP_Viewer_Client.discover_viewer_sessions", return_value=[]):
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
        with mock.patch("mcp_server.MCP_Viewer_Client.discover_viewer_sessions", return_value=[]):
            self.assertIsNone((await client.list_sessions())["connected_session_id"])
        client.connected_session_id = "closed"
        with mock.patch("mcp_server.MCP_Viewer_Client.select_viewer_session", side_effect=LookupError("gone")):
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
    def test_terminal_runner_captures_native_and_python_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(SRC / "mcp_server" / "Viewer_Terminal.py"), directory,
                sys.executable, "-u", "-c",
                "import os; print('python output'); os.write(1,b'native partial'); os.write(2,b'native error')"],
                capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((Path(directory) / "stdout.log").read_bytes(), result.stdout)
            self.assertEqual((Path(directory) / "stderr.log").read_bytes(), result.stderr)
            self.assertIn(b"python output", result.stdout)
            self.assertIn(b"native partial", result.stdout)
            self.assertIn(b"native error", result.stderr)

    def test_lost_terminal_does_not_stop_capture(self):
        output = io.BytesIO()
        terminal = mock.Mock()
        terminal.write.side_effect = OSError("closed")
        copy_output(io.BytesIO(b"x" * 20000), output, terminal)
        self.assertEqual(output.getvalue(), b"x" * 20000)


if __name__ == "__main__":
    unittest.main()
