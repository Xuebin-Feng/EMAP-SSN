import asyncio
import codecs
import io
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
from mcp_server.viewer.Viewer_Terminal import copy_output


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
    def test_terminal_runner_captures_native_and_python_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(SRC / "mcp_server" / "viewer" / "Viewer_Terminal.py"), directory,
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

    def test_viewer_prints_any_text_whatever_the_pipe_encoding(self):
        """An MCP-launched Viewer prints to Viewer_Terminal's pipe, which Python
        opens in the ANSI code page on Windows; a command echoing text outside
        it failed with UnicodeEncodeError."""
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
        ])
        # cp1252 is that pipe encoding here; the variable sets it on any platform.
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONUTF8"}
        environment["PYTHONIOENCODING"] = "cp1252"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(SRC / "mcp_server" / "viewer" / "Viewer_Terminal.py"), directory,
                sys.executable, "-u", "-c", child], capture_output=True, env=environment, timeout=60)
            log = (Path(directory) / "stdout.log").read_bytes()
            errors = (Path(directory) / "stderr.log").read_text(encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, errors)
        self.assertIn(f"Expression: {expression}".encode("utf-8"), log)
        self.assertEqual(log, result.stdout)

    def test_viewer_streams_escape_what_a_terminal_cannot_show(self):
        from EMAPSSN_Viewer import _configure_output_streams

        class Terminal(io.BytesIO):
            def isatty(self):
                return True

        pipe = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        terminal = io.TextIOWrapper(Terminal(), encoding="cp1252")
        _configure_output_streams([pipe, terminal, None])
        for stream in (pipe, terminal):
            stream.write("α")
            stream.flush()
        self.assertEqual(pipe.buffer.getvalue(), "α".encode("utf-8"))
        self.assertEqual(terminal.buffer.getvalue(), b"\\u03b1")  # a terminal keeps its encoding

    def test_terminal_copy_keeps_characters_that_a_read_cuts(self):
        class RawConsole:
            """The raw writer behind a Windows console under -u. Like any raw
            stream it may take fewer bytes than offered: here at most five a
            call, and never a character that the data cuts off at its end."""
            def __init__(self):
                self.shown = bytearray()

            def write(self, data):
                offered = bytes(data[:5])
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                decoder.decode(offered)
                taken = len(offered) - len(decoder.getstate()[0])
                self.shown += offered[:taken]
                return taken

            def flush(self):
                pass

        class Pipe:
            """Hands out CHUNKS, noting before each read what the console shows."""
            def __init__(self, chunks, console):
                self.chunks, self.console, self.shown = list(chunks), console, []

            def read(self, size):
                self.shown.append(bytes(self.console.shown))
                return self.chunks.pop(0) if self.chunks else b""

        data = "AαB中C\U0001f9eaD\n".encode("utf-8")
        reads = range(1, len(data))
        for cuts in [(i,) for i in reads] + [(i, j) for i in reads for j in reads if i < j]:
            bounds = (0, *cuts, len(data))
            chunks = [data[start:end] for start, end in zip(bounds, bounds[1:])]
            with self.subTest(chunks=chunks):
                log, console = io.BytesIO(), RawConsole()
                pipe = Pipe(chunks, console)
                copy_output(pipe, log, console)
                self.assertEqual(log.getvalue(), data)
                # Each read finds every character the earlier reads completed shown.
                whole = [codecs.getincrementaldecoder("utf-8")().decode(data[:end]).encode("utf-8")
                         for end in bounds]
                self.assertEqual(pipe.shown, whole)
        terminal = io.BytesIO()  # a POSIX terminal takes any byte, so it gets a cut-off ending too
        copy_output(io.BytesIO(data[:2]), io.BytesIO(), terminal)
        self.assertEqual(terminal.getvalue(), data[:2])


if __name__ == "__main__":
    unittest.main()
