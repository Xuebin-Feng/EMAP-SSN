import asyncio
import json
import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcp import Client, StdioServerParameters  # noqa: E402

from EMAPSSN_MCP_Server import mcp  # noqa: E402
from utilities.MCP_Pipeline_Jobs import (  # noqa: E402
    PipelineJobManager,
    PipelineQueueFullError,
)
from utilities.MCP_Viewer_Client import MCPViewerClient  # noqa: E402
from utilities.Tool_Execution import (  # noqa: E402
    ToolInvocation,
    create_settings_snapshot,
    get_tool_spec,
)
from utilities.Viewer_Sessions import SESSION_DIRECTORY_ENV  # noqa: E402


class PipelineJobManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.temporary_path = pathlib.Path(self.temporary.name)
        self.fake_script = self.temporary_path / "fake_pipeline.py"
        self.fake_script.write_text(
            """
import json
import sys
import time

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    settings = json.load(handle)["Sanitize_Sequences.py"]
time.sleep(float(settings.get("DELAY", 0)))
print(settings.get("STDOUT", ""), flush=True)
print(settings.get("STDERR", ""), file=sys.stderr, flush=True)
raise SystemExit(int(settings.get("EXIT_CODE", 0)))
""".lstrip(),
            encoding="utf-8",
        )
        self.manager = PipelineJobManager(
            PROJECT_ROOT,
            max_pending=2,
            history_limit=4,
            termination_grace=0.5,
            temporary_parent=self.temporary_path,
        )
        self.prepare_patch = mock.patch(
            "utilities.MCP_Pipeline_Jobs.prepare_headless_invocation",
            side_effect=self._prepare_fake_invocation,
        )
        self.prepare_patch.start()
        await self.manager.start()

    async def asyncTearDown(self):
        await self.manager.close()
        self.prepare_patch.stop()
        self.temporary.cleanup()

    def _document(self, **settings):
        return {
            "DIRECTORIES": {"FASTA_DIR": str(self.temporary_path / "outputs")},
            "Sanitize_Sequences.py": settings,
        }

    def _prepare_fake_invocation(
        self,
        tool_id,
        settings_source,
        project_root,
        *,
        python_executable=None,
        snapshot_directory=None,
    ):
        spec = get_tool_spec(tool_id)
        snapshot = create_settings_snapshot(
            spec,
            settings_source,
            snapshot_directory=snapshot_directory,
        )
        return ToolInvocation(
            tool=spec,
            argv=(python_executable or sys.executable, "-u", str(self.fake_script), snapshot),
            cwd=str(self.temporary_path),
            settings_path=snapshot,
            owns_settings_snapshot=True,
        )

    async def _wait_for_status(self, job_id, status, timeout=5.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            payload = await self.manager.get_job(job_id)
            if payload["status"] == status:
                return payload
            await asyncio.sleep(0.01)
        self.fail(f"Job {job_id} did not reach {status}.")

    async def test_fifo_bound_status_and_output_locations(self):
        first = await self.manager.submit(
            "sanitize_sequences",
            self._document(DELAY=0.2, STDOUT="first"),
        )
        await self._wait_for_status(first["job_id"], "running")
        second = await self.manager.submit(
            "sanitize_sequences",
            self._document(DELAY=0.01, STDOUT="second"),
        )
        third = await self.manager.submit(
            "sanitize_sequences",
            self._document(DELAY=0.01, STDOUT="third"),
        )
        self.assertEqual(second["queue_position"], 1)
        self.assertEqual(third["queue_position"], 2)
        with self.assertRaises(PipelineQueueFullError):
            await self.manager.submit(
                "sanitize_sequences",
                self._document(STDOUT="rejected"),
            )

        completed = [
            await self.manager.wait_for_terminal(item["job_id"])
            for item in (first, second, third)
        ]
        self.assertEqual([item["status"] for item in completed], ["succeeded"] * 3)
        self.assertEqual(
            completed[0]["output_locations"]["FASTA_DIR"],
            str((self.temporary_path / "outputs").resolve()),
        )
        second_log = await self.manager.read_log(
            second["job_id"],
            "stdout",
            limit=1024,
        )
        self.assertEqual(second_log["text"].strip(), "second")

    async def test_failure_and_bounded_log_paging(self):
        submitted = await self.manager.submit(
            "sanitize_sequences",
            self._document(STDOUT="abcdef", STDERR="problem", EXIT_CODE=7),
        )
        finished = await self.manager.wait_for_terminal(submitted["job_id"])
        self.assertEqual(finished["status"], "failed")
        self.assertEqual(finished["exit_code"], 7)
        first_page = await self.manager.read_log(
            submitted["job_id"], "stdout", offset=0, limit=3
        )
        second_page = await self.manager.read_log(
            submitted["job_id"],
            "stdout",
            offset=first_page["next_offset"],
            limit=1024,
        )
        self.assertEqual(first_page["text"], "abc")
        self.assertEqual(second_page["text"].replace("\r\n", "\n"), "def\n")
        self.assertTrue(second_page["eof"])
        error_log = await self.manager.read_log(
            submitted["job_id"], "stderr", limit=1024
        )
        self.assertEqual(error_log["text"].strip(), "problem")

    async def test_queued_and_running_cancellation_are_idempotent(self):
        running = await self.manager.submit(
            "sanitize_sequences",
            self._document(DELAY=10, STDOUT="never"),
        )
        await self._wait_for_status(running["job_id"], "running")
        queued = await self.manager.submit(
            "sanitize_sequences",
            self._document(STDOUT="also never"),
        )
        cancelled_queued = await self.manager.cancel(queued["job_id"])
        self.assertEqual(cancelled_queued["status"], "cancelled")
        cancelled_running = await self.manager.cancel(running["job_id"])
        self.assertEqual(cancelled_running["status"], "cancelled")
        repeated = await self.manager.cancel(running["job_id"])
        self.assertEqual(repeated["status"], "cancelled")


class MCPProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_in_process_protocol_catalog_and_validation(self):
        with tempfile.TemporaryDirectory() as session_directory, mock.patch.dict(
            os.environ,
            {SESSION_DIRECTORY_ENV: session_directory},
        ):
            async with Client(mcp) as client:
                listed = await client.list_tools()
                names = [tool.name for tool in listed.tools]
                self.assertEqual(
                    names,
                    [
                        "list_pipeline_tools",
                        "start_pipeline_job",
                        "list_pipeline_jobs",
                        "get_pipeline_job",
                        "read_pipeline_log",
                        "cancel_pipeline_job",
                        "list_viewer_sessions",
                        "get_viewer_summary",
                        "query_viewer_nodes",
                    ],
                )
                annotations = {tool.name: tool.annotations for tool in listed.tools}
                self.assertTrue(annotations["list_pipeline_tools"].read_only_hint)
                self.assertTrue(annotations["start_pipeline_job"].destructive_hint)
                self.assertTrue(annotations["cancel_pipeline_job"].idempotent_hint)

                catalog = await client.call_tool("list_pipeline_tools")
                self.assertFalse(catalog.is_error)
                self.assertEqual(len(catalog.structured_content["tools"]), 14)
                self.assertEqual(catalog.structured_content["max_pending"], 16)

                invalid = await client.call_tool(
                    "start_pipeline_job",
                    {"tool_id": "sanitize_sequences"},
                )
                self.assertTrue(invalid.is_error)
                viewers = await client.call_tool("list_viewer_sessions")
                self.assertEqual(
                    viewers.structured_content,
                    {"sessions": [], "automatic_selection": False},
                )

    async def test_real_stdio_subprocess_has_a_clean_protocol_channel(self):
        with tempfile.TemporaryDirectory() as session_directory:
            environment = dict(os.environ)
            environment[SESSION_DIRECTORY_ENV] = session_directory
            parameters = StdioServerParameters(
                command=sys.executable,
                args=[str(SRC_DIR / "EMAPSSN_MCP_Server.py")],
                cwd=PROJECT_ROOT,
                env=environment,
            )
            async with Client(parameters, read_timeout_seconds=10) as client:
                listed = await client.list_tools()
                self.assertEqual(len(listed.tools), 9)
                catalog = await client.call_tool("list_pipeline_tools")
                self.assertFalse(catalog.is_error)
                self.assertEqual(len(catalog.structured_content["tools"]), 14)


class MCPViewerClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_listing_does_not_disclose_secrets(self):
        descriptor = SimpleNamespace(
            session_id="viewer-one",
            pid=123,
            started_at="2026-08-28T00:00:00Z",
            token="secret-token",
            descriptor_path="secret-path",
        )
        with mock.patch(
            "utilities.MCP_Viewer_Client.discover_viewer_sessions",
            return_value=[descriptor],
        ):
            payload = await MCPViewerClient().list_sessions()
        self.assertEqual(
            payload,
            {
                "sessions": [
                    {
                        "session_id": "viewer-one",
                        "pid": 123,
                        "started_at": "2026-08-28T00:00:00Z",
                    }
                ],
                "automatic_selection": True,
            },
        )
        self.assertNotIn("secret-token", json.dumps(payload))
        self.assertNotIn("secret-path", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
