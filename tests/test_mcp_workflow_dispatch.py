"""Workflow permission boundaries, discovery and argument dispatch contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from typing import get_args
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError
from EMAPSSN_MCP_Server import mcp
from mcp_server.core import App_Context as app_ctx
from mcp_server.core.Workflow_Dispatch import (
    REGISTRY, PipelineAction, ViewerDataAction, ViewerControlAction, dispatch,
)
from mcp_server.pipeline import Pipeline_Operations as pipeline_ops
from mcp_server.viewer import Viewer_Operations as viewer_ops


class WorkflowDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_catalog_discovery_and_read_only_dispatch(self):
        from desktop.Viewer_Inspection import command_catalog
        async def read_catalog(ctx, action, arguments, session_id):
            self.assertEqual(action, 'get_command_catalog')
            return command_catalog(**arguments)
        with mock.patch.object(viewer_ops, '_portal_call', side_effect=read_catalog):
            general = await dispatch('emapssn_viewer_data', 'get_command_catalog', {}, None)
            detailed = await dispatch('emapssn_viewer_data', 'get_command_catalog', {'command': 'reset'}, None)
        self.assertTrue(all(c['help'] is None for c in general['commands']))
        self.assertIn('reset <TARGET_1> [TARGET_2] ...', detailed['commands'][0]['syntax'])
        description = await dispatch('emapssn_viewer_data', 'describe', {'action': 'get_command_catalog'}, None)
        self.assertEqual(description['example']['arguments'], {'command': 'reset'})

    async def test_every_action_discovers_and_forwards_validated_arguments(self):
        ctx = object()
        for workflow, actions in REGISTRY.items():
            help_result = await dispatch(workflow, "help", {}, ctx)
            self.assertEqual({item["action"] for item in help_result["actions"]},
                             set(actions) | {"help", "describe"})
            for name, entry in actions.items():
                with self.subTest(workflow=workflow, action=name):
                    description = await dispatch(workflow, "describe", {"action": name}, ctx)
                    schema = description["arguments_schema"]
                    self.assertFalse(schema["additionalProperties"])
                    self.assertNotIn("ctx", schema["properties"])
                    self.assertTrue(description["effects"])
                    self.assertNotIn("export_config_settings", json.dumps(description))
                    self.assertNotIn("action='emapssn_", json.dumps(description))
                    example = description["example"]["arguments"]
                    expected = entry.model.model_validate(example).model_dump()
                    if entry.needs_context:
                        expected["ctx"] = ctx
                    result = {"unchanged_payload": [1, None, {"nested": True}]}
                    with mock.patch.object(entry.module, entry.handler_name,
                                           new_callable=mock.AsyncMock, return_value=result) as handler:
                        self.assertEqual(await dispatch(workflow, name, example, ctx), result)
                        handler.assert_awaited_once_with(**expected)

    async def test_invalid_requests_never_reach_any_handler(self):
        for workflow, actions in REGISTRY.items():
            for name, entry in actions.items():
                with self.subTest(workflow=workflow, action=name):
                    with mock.patch.object(entry.module, entry.handler_name) as handler:
                        with self.assertRaises(ToolError):
                            await dispatch(workflow, name, {**entry.example, "unexpected": True}, None)
                        handler.assert_not_called()
        invalid = [
            ("emapssn_pipeline", "help", {"unexpected": True}),
            ("emapssn_pipeline", "describe", {"action": 1}),
            ("emapssn_pipeline", "describe", {"action": "missing"}),
            ("emapssn_pipeline", "missing", {}),
            ("emapssn_pipeline", "get_job", {}),
            ("emapssn_pipeline", "start_job", {"tool_id": 7}),
            ("emapssn_pipeline", "list_jobs", {"limit": "10"}),
            ("emapssn_pipeline", "list_jobs", {"limit": True}),
            ("emapssn_pipeline", "list_jobs", {"limit": 101}),
            ("emapssn_pipeline", "start_layout_job", {"umap_mode": "false"}),
            ("emapssn_pipeline", "export_layout_settings", {"kind": "viewer"}),
            ("emapssn_viewer_control", "export_settings", {"kind": "layout"}),
            ("emapssn_viewer_data", "query_nodes", {"columns": [1]}),
            ("emapssn_viewer_data", "query_nodes", {"limit": 501}),
            ("emapssn_viewer_data", "query_nodes", {"scope": "invalid"}),
            ("emapssn_viewer_data", "close_session", {}),
        ]
        with mock.patch.object(app_ctx, "_context") as context, mock.patch.object(app_ctx, "_viewer") as viewer:
            for workflow, name, args in invalid:
                with self.subTest(workflow=workflow, action=name, arguments=args):
                    with self.assertRaises(ToolError):
                        await dispatch(workflow, name, args, None)
            context.assert_not_called()
            viewer.assert_not_called()

    async def test_split_exports_delegate_correct_kind(self):
        with mock.patch("utilities.Headless_Settings.export_config_settings",
                        return_value={"exported": True}) as export:
            for workflow, action, kind in [
                ("emapssn_pipeline", "export_layout_settings", "layout"),
                ("emapssn_viewer_control", "export_settings", "viewer"),
            ]:
                result = await dispatch(workflow, action, {"output_path": "out.json", "settings_path": "overlay.json"}, None)
                self.assertEqual(result, {"exported": True})
                export.assert_called_with(kind, pipeline_ops._PROJECT_ROOT, "out.json", "overlay.json")

    async def test_models_and_errors_keep_original_payload(self):
        with mock.patch.object(pipeline_ops, "list_pipeline_jobs", new_callable=mock.AsyncMock,
                               return_value=pipeline_ops.PipelineJobList(jobs=[])):
            self.assertEqual(await dispatch("emapssn_pipeline", "list_jobs", {}, None), {"jobs": []})
        with mock.patch.object(viewer_ops, "get_viewer_summary", side_effect=ToolError("Viewer unavailable")):
            with self.assertRaisesRegex(ToolError, "Viewer unavailable"):
                await dispatch("emapssn_viewer_data", "get_summary", {}, None)

    async def test_layout_settings_path_uses_existing_queue_and_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            document = {"schema_version": 2, "kind": "layout"}
            path.write_text(json.dumps(document), encoding="utf-8")
            settings = object()
            jobs = SimpleNamespace(submit_layout_job=mock.AsyncMock(return_value={"job_id": "layout-job"}))
            ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=SimpleNamespace(jobs=jobs)))
            with mock.patch("Layout_Cache_Generator.LayoutGenerationSettings.from_document", return_value=settings) as parse, \
                    mock.patch.object(pipeline_ops, "_job_info", side_effect=lambda value: value):
                result = await dispatch("emapssn_pipeline", "start_layout_job", {"settings_path": str(path)}, ctx)
            self.assertEqual(result, {"job_id": "layout-job"})
            parse.assert_called_once_with(document, project_root=pipeline_ops._PROJECT_ROOT)
            jobs.submit_layout_job.assert_awaited_once_with(settings)

    async def test_protocol_enums_defaults_discovery_and_old_names_removed(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(tempfile, "tempdir", directory):
            async with Client(mcp) as client:
                catalog = await client.list_tools()
                self.assertEqual({tool.name for tool in catalog.tools}, set(REGISTRY))
                for tool, enum in zip(catalog.tools, [PipelineAction, ViewerDataAction, ViewerControlAction]):
                    self.assertEqual(set(tool.input_schema["properties"]["action"]["enum"]), set(get_args(enum)))
                    self.assertEqual(set(get_args(enum)), set(REGISTRY[tool.name]) | {"help", "describe"})
                    self.assertEqual(tool.input_schema["required"], ["action"])
                    result = await client.call_tool(tool.name, {"action": "help"})
                    self.assertFalse(result.is_error)
                    for name in ("help", "describe", *REGISTRY[tool.name]):
                        result = await client.call_tool(tool.name, {"action": "describe", "arguments": {"action": name}})
                        self.assertFalse(result.is_error)
                old_names = {entry.handler_name for actions in REGISTRY.values() for entry in actions.values()}
                old_names.add("export_config_settings")
                for name in old_names:
                    self.assertTrue((await client.call_tool(name, {})).is_error)
                for args in ({"action": "missing"}, {"action": "help", "arguments": None},
                             {"action": "help", "arguments": []}):
                    self.assertTrue((await client.call_tool("emapssn_pipeline", args)).is_error)


if __name__ == "__main__":
    unittest.main()
