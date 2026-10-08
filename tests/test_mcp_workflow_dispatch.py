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
from mcp_server.viewer import Viewer_Operations as ops


class WorkflowDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_catalog_discovery_and_read_only_dispatch(self):
        from desktop.Viewer_Inspection import command_catalog
        async def read_catalog(ctx, action, arguments, session_id):
            self.assertEqual(action, 'get_command_catalog')
            return command_catalog(**arguments)
        connected = SimpleNamespace(connected_session_id='viewer')
        with mock.patch.object(viewer_ops, '_portal_call', side_effect=read_catalog), \
                mock.patch.object(viewer_ops, '_viewer', return_value=connected):
            general = await dispatch('emapssn_viewer_data', 'get_command_catalog', {}, None)
            detailed = await dispatch('emapssn_viewer_data', 'get_command_catalog', {'command': 'reset'}, None)
        self.assertTrue(all(c['help'] is None for c in general['commands']))
        self.assertIn('reset <TARGET_1> [TARGET_2] ...', detailed['commands'][0]['syntax'])
        self.assertNotIn('source', general)
        # Without a Viewer the static catalog comes from the command sources.
        disconnected = SimpleNamespace(connected_session_id=None)
        with mock.patch.object(viewer_ops, '_portal_call') as portal, \
                mock.patch.object(viewer_ops, '_viewer', return_value=disconnected):
            local = await dispatch('emapssn_viewer_data', 'get_command_catalog', {}, None)
            local_reset = await dispatch('emapssn_viewer_data', 'get_command_catalog', {'command': 'reset'}, None)
            with self.assertRaisesRegex(ToolError, 'Unknown command'):
                await dispatch('emapssn_viewer_data', 'get_command_catalog', {'command': 'missing'}, None)
            routed = await dispatch('emapssn_viewer_data', 'get_command_catalog', {'session_id': 'explicit'}, None)
        self.assertEqual(portal.call_count, 1)  # Only the explicit session_id reached a Viewer.
        self.assertIs(routed, portal.return_value)
        self.assertEqual(local, {**general, 'source': 'installation'})
        self.assertEqual(local_reset['commands'], detailed['commands'])
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
            for workflow, action, expected in [
                # A layout export without inline inputs or fields passes no overlay.
                ("emapssn_pipeline", "export_layout_settings", ("layout", pipeline_ops._PROJECT_ROOT, "out.json", "overlay.json", None)),
                ("emapssn_viewer_control", "export_settings", ("viewer", pipeline_ops._PROJECT_ROOT, "out.json", "overlay.json")),
            ]:
                result = await dispatch(workflow, action, {"output_path": "out.json", "settings_path": "overlay.json"}, None)
                self.assertEqual(result, {"exported": True})
                export.assert_called_with(*expected)
            await dispatch("emapssn_pipeline", "export_layout_settings", {
                "node_fasta_file": "nodes.fasta", "input_hdf5": "net.h5",
                "parameters": {"top_edge_percent": 5, "dt": 0.002}}, None)
            overlay = export.call_args.args[4]
            self.assertEqual(overlay, {"schema_version": 2, "kind": "layout",
                                       "inputs": {"NODE_FASTA_FILE": "nodes.fasta", "INPUT_HDF5": "net.h5"},
                                       "network": {"TOP_EDGE_PERCENT": 5}, "simulation": {"DT": 0.002}})

    async def test_viewer_export_reports_saved_state_of_each_cache(self):
        # Export picks the newest cache, which may lack the clusters a saved
        # figure cache holds; the result must make that choice visible.
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            def cache(name, clusters=None):
                with h5py.File(folder / name, "w") as hf:
                    hf.create_dataset("headers", data=[b"A", b"B"])
                    hf.create_dataset("positions", data=np.zeros((2, 2), np.float32))
                    if clusters is not None:
                        hf.create_dataset("cluster_labels", data=clusters)
                        hf.attrs["last_cluster_params"] = json.dumps(["LEIDEN_1.0", 20])
            cache("Figure_1.h5", [0, -1])
            cache("version_01.h5")
            (folder / "copy.fasta").write_text(">A\nAC\n")
            selected = str(folder / "version_01.h5")
            with mock.patch("utilities.Headless_Settings.export_config_settings",
                            return_value={"settings_path": "export.json", "cache_path": selected}):
                result = await dispatch("emapssn_viewer_control", "export_settings", {}, None)
        self.assertEqual(result["cache_path"], selected)
        self.assertEqual(result["cache_contents"]["datasets"], ["headers", "positions"])
        self.assertIsNone(result["cache_contents"]["cluster_labels"])
        self.assertEqual([entry["cache_filename"] for entry in result["other_caches"]], ["Figure_1.h5"])
        self.assertEqual(result["other_caches"][0]["cluster_labels"], {"clusters": 1, "noise_nodes": 1})
        self.assertEqual(result["other_caches"][0]["last_cluster_params"], ["LEIDEN_1.0", 20])

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


class MCPFollowupTests(unittest.IsolatedAsyncioTestCase):
    async def test_lookup_validation_and_schema_before_transport(self):
        import jsonschema
        for action in ('get_command_request', 'read_command_output'):
            model = REGISTRY['emapssn_viewer_data'][action].model
            schema = model.model_json_schema()
            invalid = [{}, {'request_id': 'r', 'submission_id': 's'}, {'request_id': ''},
                       {'submission_id': ' '}, {'request_id': 1}, {'submission_id': False}]
            with mock.patch.object(ops, '_portal_call') as transport:
                for arguments in invalid:
                    with self.subTest(action=action, arguments=arguments):
                        with self.assertRaises(ToolError):
                            await dispatch('emapssn_viewer_data', action, arguments, None)
                        with self.assertRaises(jsonschema.ValidationError):
                            jsonschema.validate(arguments, schema)
                transport.assert_not_called()
            for arguments in ({'request_id': 'r'}, {'submission_id': 's'}, {'request_id': None, 'submission_id': 's'}):
                jsonschema.validate(arguments, schema)
                with mock.patch.object(ops, '_portal_call', new_callable=mock.AsyncMock, return_value={}) as transport:
                    await dispatch('emapssn_viewer_data', action, arguments, None)
                    forwarded = transport.call_args.args[2]
                    for key, value in arguments.items():
                        self.assertEqual(forwarded[key], value)

    async def test_execution_followup_uses_returned_session_on_new_and_retry(self):
        result = {'request_id': 'r', 'session_id': 'origin', 'submission_id': 's', 'status': 'queued'}
        with mock.patch.object(ops, '_portal_call', new_callable=mock.AsyncMock, return_value=result):
            for _ in range(2):
                response = await dispatch('emapssn_viewer_control', 'execute_commands',
                                          {'submission_id': 's', 'commands': 'select help'}, None)
                step = response['next_step']
                self.assertEqual(step, {'tool': 'emapssn_viewer_data', 'action': 'get_command_request',
                                        'arguments': {'request_id': 'r', 'session_id': 'origin'}})
                REGISTRY[step['tool']][step['action']].model.model_validate(step['arguments'])
                self.assertNotIn('next_step', result)
        self.assertNotIn('execute_commands', REGISTRY['emapssn_viewer_data'])

    async def test_session_selection_and_reconnect_keep_lookup_local(self):
        from mcp_server.viewer.Viewer_Client import MCPViewerClient
        # Exercise the actual client routing while keeping transport read-only.
        client = MCPViewerClient(Path.cwd())
        selected = []
        def discover(target, **kwargs):
            selected.append(target)
            return SimpleNamespace(session_id=target)
        def request(session, endpoint, payload=None):
            if endpoint.endswith('/session'):
                return {'inspection_capabilities': ['commands_v1']}
            self.assertEqual(payload['arguments'], {'submission_id': 's'})
            return {'session_id': session.session_id, 'request_id': session.session_id + '-r'}
        with mock.patch('mcp_server.viewer.Viewer_Client.select_viewer_session', side_effect=discover), \
                mock.patch.object(client, '_request', side_effect=request):
            client.connected_session_id = 'first'
            original = await client.command_action('get_command_request', {'submission_id': 's'})
            client.connected_session_id = 'second'
            other = await client.command_action('get_command_request', {'submission_id': 's'})
            explicit = await client.command_action('get_command_request', {'submission_id': 's'}, 'first')
            client.connected_session_id = None
            client.connected_session_id = 'first'
            reconnected = await client.command_action('get_command_request', {'submission_id': 's'})
        self.assertEqual(original, explicit)
        self.assertEqual(original, reconnected)
        self.assertNotEqual(original, other)
        self.assertEqual(selected, ['first', 'second', 'first', 'first'])


if __name__ == "__main__":
    unittest.main()
