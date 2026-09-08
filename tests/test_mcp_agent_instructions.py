"""Agent guidance reaches clients without changing or executing scientific tools."""
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from EMAPSSN_MCP_Server import _load_agent_instructions, mcp
from mcp_server.Workflow_Dispatch import REGISTRY
from mcp import Client, StdioServerParameters

GUIDE = SRC / "mcp_server" / "Agent_Instructions.md"


class GuideLoadingTests(unittest.TestCase):
    def test_exact_utf8_contents_and_working_directory_independence(self):
        expected = GUIDE.read_text(encoding="utf-8")
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(_load_agent_instructions(), expected)
            finally:
                os.chdir(original)
        self.assertGreaterEqual(len(expected.split()), 800)
        self.assertLessEqual(len(expected.split()), 1500)

    def test_missing_unreadable_or_invalid_utf8_guide_is_actionable(self):
        for error in (FileNotFoundError(), PermissionError(),
                      UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")):
            with self.subTest(error=type(error).__name__), mock.patch.object(Path, "read_text", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "Agent_Instructions.md.*restart the server"):
                    _load_agent_instructions()

    def test_empty_guide_is_actionable(self):
        with mock.patch.object(Path, "read_text", return_value=" \n\t"):
            with self.assertRaisesRegex(RuntimeError, "is empty.*Restore Agent_Instructions.md"):
                _load_agent_instructions()


class GuideProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.patch = mock.patch.object(tempfile, "tempdir", self.directory.name)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    async def test_in_process_instructions_and_complete_navigation(self):
        expected = GUIDE.read_text(encoding="utf-8")
        async with Client(mcp) as client:
            self.assertEqual(client.instructions, expected)
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            table = expected.split("## Run a pipeline")[0]
            references = set(re.findall(r"`([a-z]+_[a-z_]+)`", table))
            self.assertEqual(references, names)
            self.assertEqual(len(names), 3)
            # Every standalone tool reference in the prose must also resolve.
            identifiers = set(re.findall(r"`([a-z]+_[a-z_]+)`", expected))
            fields = {"tool_id", "settings_document", "settings_path", "job_id",
                      "failure_message", "next_offset"}
            self.assertFalse(identifiers - names - fields)
            action_references = re.findall(r'(emapssn_[a-z_]+)\(action="([a-z_]+)"\)', expected)
            self.assertTrue(action_references)
            for workflow, action in action_references:
                self.assertIn(workflow, REGISTRY)
                self.assertIn(action, REGISTRY[workflow])
            for tool in listed.tools:
                self.assertTrue(tool.description, tool.name)

    async def test_real_stdio_instructions_from_unrelated_directory(self):
        environment = dict(os.environ, TEMP=self.directory.name, TMP=self.directory.name,
                           SSN_VIEWER_SESSION_DIR=self.directory.name, PYTHONIOENCODING="utf-8")
        parameters = StdioServerParameters(command=sys.executable,
            args=[str(SRC / "EMAPSSN_MCP_Server.py")], cwd=self.directory.name, env=environment)
        # Exercise both explicit initialization and the SDK's default discovery.
        for mode in ("legacy", "auto"):
            with self.subTest(mode=mode):
                async with Client(parameters, mode=mode, read_timeout_seconds=30) as client:
                    self.assertEqual(client.instructions, GUIDE.read_text(encoding="utf-8"))
                    self.assertEqual({tool.name for tool in (await client.list_tools()).tools}, set(REGISTRY))
                    self.assertTrue((await client.call_tool("list_pipeline_tools", {})).is_error)
                    for workflow in REGISTRY:
                        self.assertFalse((await client.call_tool(workflow, {"action": "help"})).is_error)


if __name__ == "__main__":
    unittest.main()
