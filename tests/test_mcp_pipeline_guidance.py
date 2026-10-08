"""Planning knowledge, job waiting and output reporting, network statistics and the layout contract."""
import asyncio
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.pipeline import Pipeline_Guide  # noqa: E402
from mcp_server.pipeline.Pipeline_Jobs import (  # noqa: E402
    PipelineJobManager, changed_files, directory_snapshot, latest_line,
)
from tools.tool_helpers.Tool_Pipeline import (  # noqa: E402
    ToolInvocation, create_settings_snapshot, get_tool_spec, list_tool_specs,
)


def write_alignment_network(path, headers, pairs):
    """pairs: (i, j, global score); alignment lengths are 1 so scores stay unnormalized."""
    with h5py.File(path, "w") as network:
        network.attrs["model_name"] = "test_model"
        network.create_dataset("headers", data=np.asarray(headers, dtype=object),
                               dtype=h5py.string_dtype("utf-8"))
        network.create_dataset("i", data=np.asarray([p[0] for p in pairs], dtype=np.uint16))
        network.create_dataset("j", data=np.asarray([p[1] for p in pairs], dtype=np.uint16))
        network.create_dataset("seq_lens", data=np.full(len(headers), 4, dtype=np.uint16))
        scores = np.asarray([p[2] for p in pairs], dtype=np.float32)
        for name in ("g_score", "l_score"):
            network.create_dataset(name, data=scores)
        for name in ("g_len", "l_len"):
            network.create_dataset(name, data=np.ones(len(pairs), dtype=np.uint16))


def write_fasta(path, headers):
    residues = "ACDEFGHIKLMNPQRSTVWY"
    with open(path, "w", encoding="utf-8") as handle:
        for index, header in enumerate(headers):
            handle.write(f">{header}\n{residues[index % 20] * 3}{residues[(index + 7) % 20]}\n")


class PipelineGuideTests(unittest.TestCase):
    def test_every_tool_has_consistent_planning_knowledge(self):
        from mcp_server.pipeline.Pipeline_Settings import _CONTRACTS
        actions = {"start_layout_job", "network_statistics", "export_layout_settings"}
        tool_ids = {spec.tool_id for spec in list_tool_specs()}
        self.assertEqual(set(Pipeline_Guide._GUIDE["tools"]), tool_ids)
        for spec in list_tool_specs():
            with self.subTest(tool=spec.tool_id):
                guide = Pipeline_Guide.tool_guide(spec.tool_id)
                self.assertEqual(set(guide), set(Pipeline_Guide.TOOL_GUIDE_FIELDS))
                self.assertTrue(guide["stage"] and guide["purpose"] and guide["outputs"])
                properties = _CONTRACTS[spec.script_name]["properties"]
                for field, entry in guide["inputs"].items():
                    self.assertIn(field, properties)
                    self.assertIn(entry["directory"], spec.required_directories)
                for output in guide["outputs"]:
                    self.assertTrue(output["name"] and output["existing"])
                    if "directory" in output:
                        self.assertIn(output["directory"], spec.required_directories)
                    else:
                        self.assertIn(output["beside"], guide["inputs"])
                for step in guide["next"]:
                    self.assertIn(step.split()[0], tool_ids | actions | {"Viewer"}, step)
        for workflow in Pipeline_Guide.workflows():
            self.assertTrue(workflow["goal"] and workflow["steps"])

    def test_layout_call_examples_match_the_action_arguments(self):
        from mcp_server.core.Workflow_Dispatch import REGISTRY
        from mcp_server.pipeline.Pipeline_Operations import LAYOUT_CALL_EXAMPLES, list_pipeline_tools
        self.assertEqual(list_pipeline_tools().layout["examples"], LAYOUT_CALL_EXAMPLES)
        for example in LAYOUT_CALL_EXAMPLES:
            with self.subTest(action=example["action"]):
                REGISTRY["emapssn_pipeline"][example["action"]].model.model_validate(example["arguments"])

    def test_every_bundled_model_has_an_embedding_width(self):
        from tools.tool_helpers.Model_Plugins import discover_model_execution_modes
        models = discover_model_execution_modes(str(SRC / "resources" / "pLM_models"))
        self.assertEqual(set(models), set(Pipeline_Guide._GUIDE["model_embedding_widths"]))
        self.assertEqual(Pipeline_Guide.embedding_width("esm2_t33_650m"), 1280)
        self.assertIsNone(Pipeline_Guide.embedding_width("unlisted_plugin_model"))

    def test_input_paths_and_watched_folders_follow_each_tool(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            elsewhere = root / "elsewhere" / "raw.fasta"
            document = {"DIRECTORIES": {"FASTA_DIR": str(root / "fasta")},
                        "Sanitize_Sequences.py": {"INPUT_FASTA": str(elsewhere)}}
            self.assertEqual(Pipeline_Guide.input_path("sanitize_sequences", "INPUT_FASTA", document, root),
                             str(elsewhere))
            self.assertEqual(Pipeline_Guide.watched_directories("sanitize_sequences", document, root),
                             (str(root / "fasta"), str(root / "elsewhere")))
            document["Sanitize_Sequences.py"]["INPUT_FASTA"] = "raw.fasta"
            self.assertEqual(Pipeline_Guide.input_path("sanitize_sequences", "INPUT_FASTA", document, root),
                             str(root / "fasta" / "raw.fasta"))
            self.assertEqual(Pipeline_Guide.watched_directories("sanitize_sequences", document, root),
                             (str(root / "fasta"),))
            # Missing directories fall back to the project defaults, like the tools.
            converter = {"DIRECTORIES": {}, "Sparse_MSA_Converter.py": {"CONVERT_ALL": True, "INPUT_FASTA": None}}
            self.assertEqual(Pipeline_Guide.watched_directories("sparse_msa_converter", converter, root),
                             (str(root / "Input_Files" / "Multiple_Alignments"),))

    def test_tool_schema_carries_guide_and_model_availability(self):
        from mcp_server.pipeline.Pipeline_Settings import get_pipeline_schema
        schema = get_pipeline_schema("embedding_injection", ROOT)
        self.assertIn("never modified", schema["outputs"][0]["existing"])
        self.assertEqual(schema["inputs"]["INPUT_EMBED"]["directory"], "EMBED_DIR")
        self.assertNotIn("model_availability", schema)
        self.assertIn("model_availability", get_pipeline_schema("generate_embeddings", ROOT))
        self.assertIn("saved Tools directory", json.dumps(schema["directories_schema"]))


class ModelAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        plugins = self.root / "src" / "resources" / "pLM_models"
        shutil.copytree(SRC / "resources" / "pLM_models", plugins,
                        ignore=shutil.ignore_patterns("*.json", "__pycache__"))

    def tearDown(self):
        self.temporary.cleanup()

    def test_jobs_cannot_prompt_so_missing_credentials_and_licenses_are_reported(self):
        from mcp_server.pipeline import Pipeline_Settings
        with mock.patch.dict(os.environ, {"ESM_API_KEY": ""}), \
                mock.patch.object(Pipeline_Settings, "is_model_license_accepted", return_value=False):
            report = Pipeline_Settings.model_availability(self.root)
        self.assertTrue(report["esm2_t33_650m"]["usable_in_mcp_jobs"])
        self.assertIsNone(report["esm2_t33_650m"]["action_needed"])
        self.assertFalse(report["esmc_6b"]["usable_in_mcp_jobs"])
        self.assertFalse(report["esmc_6b"]["credentials_stored"])
        self.assertFalse(report["ankh_base"]["usable_in_mcp_jobs"])
        self.assertIn("--accept-model-license ankh_base", report["ankh_base"]["action_needed"])

        (self.root / "src" / "resources" / "Biohub_API.json").write_text('{"ESM_API_TOKEN": "secret-value"}')
        with mock.patch.object(Pipeline_Settings, "is_model_license_accepted", return_value=True):
            report = Pipeline_Settings.model_availability(self.root)
        self.assertTrue(report["esmc_6b"]["usable_in_mcp_jobs"])
        self.assertTrue(report["ankh_large"]["license_acknowledged"])
        self.assertNotIn("secret-value", json.dumps(report))


class OutputReportingTests(unittest.TestCase):
    def test_snapshot_diff_reports_created_modified_and_deleted_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = pathlib.Path(temp)
            (folder / "kept.txt").write_text("a")
            (folder / "changed.txt").write_text("a")
            (folder / "removed.txt").write_text("a")
            (folder / "sub").mkdir()
            before = directory_snapshot([folder, folder / "missing"])
            (folder / "changed.txt").write_text("longer")
            (folder / "removed.txt").unlink()
            (folder / "new.h5").write_bytes(b"12345")
            (folder / "new.h5.partial").write_bytes(b"x")
            after = directory_snapshot([folder])
            changes = {pathlib.Path(item["path"]).name: (item["change"], item["size_bytes"])
                       for item in changed_files(before, after)}
        self.assertEqual(changes, {"changed.txt": ("modified", 6), "new.h5": ("created", 5),
                                   "removed.txt": ("deleted", None)})

    def test_depth_two_covers_cache_folders_but_not_locks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "old_cache").mkdir()
            (root / "old_cache" / "version_00.h5").write_bytes(b"1")
            before = directory_snapshot([root], depth=2)
            (root / ".layout-generation.lock").write_bytes(b"0")
            (root / "new_cache" / "deeper").mkdir(parents=True)
            (root / "new_cache" / "version_00.h5").write_bytes(b"12")
            (root / "new_cache" / "cache_manifest.json").write_text("{}")
            (root / "new_cache" / "deeper" / "ignored.txt").write_text("x")
            changes = changed_files(before, directory_snapshot([root], depth=2))
            self.assertEqual(sorted(pathlib.Path(item["path"]).relative_to(root).as_posix() for item in changes),
                             ["new_cache/cache_manifest.json", "new_cache/version_00.h5"])
            self.assertEqual(directory_snapshot([root]), {})

    def test_latest_line_shows_progress_redraws_and_long_tails(self):
        with tempfile.TemporaryDirectory() as temp:
            log = pathlib.Path(temp) / "stderr.log"
            log.write_bytes(b"start\nAligning: 10%|#\rAligning: 55%|#####\r\n\n")
            self.assertEqual(latest_line(log), "Aligning: 55%|#####")
            log.write_bytes(b"x" * 10000 + "\nDone ✅\n".encode())
            self.assertEqual(latest_line(log), "Done ✅")
            log.write_bytes(b"")
            self.assertIsNone(latest_line(log))
            self.assertIsNone(latest_line(pathlib.Path(temp) / "missing.log"))


class JobWaitingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = pathlib.Path(self.temporary.name)
        self.script = self.path / "fake_tool.py"
        self.script.write_text(
            "import json, sys, time, pathlib\n"
            "document = json.load(open(sys.argv[1], encoding='utf-8'))\n"
            "settings = document['Sanitize_Sequences.py']\n"
            "time.sleep(float(settings.get('DELAY', 0)))\n"
            "if settings.get('WRITE'):\n"
            "    pathlib.Path(document['DIRECTORIES']['FASTA_DIR'], settings['WRITE']).write_text('>a\\nAC\\n')\n"
            "print('working', flush=True)\n"
            "print('progress 50%\\rprogress 100%', file=sys.stderr, flush=True)\n",
            encoding="utf-8",
        )
        self.manager = PipelineJobManager(ROOT, termination_grace=0.5, temporary_parent=self.path)
        self.patch = mock.patch("mcp_server.pipeline.Pipeline_Jobs.prepare_headless_invocation",
                                side_effect=self._invocation)
        self.patch.start()
        await self.manager.start()

    async def asyncTearDown(self):
        await self.manager.close()
        self.patch.stop()
        self.temporary.cleanup()

    def _invocation(self, tool_id, settings_source, project_root, *, python_executable=None,
                    snapshot_directory=None):
        spec = get_tool_spec(tool_id)
        snapshot = create_settings_snapshot(spec, settings_source, snapshot_directory=snapshot_directory)
        return ToolInvocation(tool=spec, argv=(sys.executable, "-u", str(self.script), snapshot),
                              cwd=str(self.path), settings_path=snapshot, owns_settings_snapshot=True)

    def _document(self, **settings):
        outputs = self.path / "outputs"
        outputs.mkdir(exist_ok=True)
        return {"DIRECTORIES": {"FASTA_DIR": str(outputs)}, "Sanitize_Sequences.py": settings}

    async def test_wait_returns_on_completion_with_output_files_and_progress(self):
        job = await self.manager.submit("sanitize_sequences", self._document(DELAY=0.3, WRITE="made.fasta"))
        started = time.monotonic()
        finished = await self.manager.wait(job["job_id"], timeout=30)
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual([(pathlib.Path(f["path"]).name, f["change"]) for f in finished["output_files"]],
                         [("made.fasta", "created")])
        self.assertEqual(finished["latest_output"], {"stdout": "working", "stderr": "progress 100%"})
        listed = await self.manager.list_jobs()
        self.assertEqual(listed[0]["output_files"], finished["output_files"])
        self.assertNotIn("latest_output", listed[0])

    async def test_wait_times_out_with_the_running_state(self):
        job = await self.manager.submit("sanitize_sequences", self._document(DELAY=5))
        waited = await self.manager.wait(job["job_id"], timeout=0.3)
        self.assertIn(waited["status"], {"queued", "running"})
        self.assertEqual(waited["output_files"], [])
        await self.manager.cancel(job["job_id"])

    async def test_a_job_that_writes_nothing_reports_no_files(self):
        job = await self.manager.submit("sanitize_sequences", self._document())
        finished = await self.manager.wait(job["job_id"], timeout=30)
        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual(finished["output_files"], [])

    async def test_waiting_through_mcp_does_not_block_other_calls(self):
        from mcp import Client
        from EMAPSSN_MCP_Server import mcp
        with mock.patch("mcp_server.pipeline.Pipeline_Operations._context",
                        return_value=SimpleNamespace(jobs=self.manager)), \
                mock.patch("mcp_server.pipeline.Pipeline_Jobs.tempfile.gettempdir", return_value=str(self.path)):
            async with Client(mcp, read_timeout_seconds=60) as client:
                job = await self.manager.submit("sanitize_sequences", self._document(DELAY=2))
                waiting = asyncio.create_task(client.call_tool(
                    "emapssn_pipeline", {"action": "wait_job", "arguments": {"job_id": job["job_id"]}}))
                await asyncio.sleep(0.2)
                status = await asyncio.wait_for(client.call_tool(
                    "emapssn_pipeline", {"action": "get_job", "arguments": {"job_id": job["job_id"]}}), 5)
                self.assertFalse(waiting.done())
                self.assertIn(status.structured_content["status"], {"queued", "running"})
                finished = await asyncio.wait_for(waiting, 30)
        self.assertFalse(finished.is_error, finished.content)
        self.assertEqual(finished.structured_content["status"], "succeeded")

    async def test_layout_results_record_the_cache_summary_and_reject_bad_results(self):
        job_directory = self.path / "layout-job"
        job_directory.mkdir()
        cache = self.path / "cache.h5"
        cache.write_bytes(b"12")
        job = SimpleNamespace(invocation=SimpleNamespace(settings_path=str(job_directory / "layout.json")),
                              output_locations={"SAVED_LAYOUT_DIR": str(self.path)}, result=None)
        result = {"TARGET_CACHE_PATH": str(cache), "CACHE_FILENAME": "cache.h5", "SAVED_LAYOUT_DIR": str(self.path),
                  "summary": {"nodes": 2, "clusters": 1}}
        (job_directory / "layout-result.json").write_text(json.dumps(result))
        entry = PipelineJobManager._read_layout_result(job)
        self.assertEqual(entry, {"path": str(cache), "change": "created", "size_bytes": 2})
        self.assertEqual(job.result, {"nodes": 2, "clusters": 1})
        self.assertNotIn("summary", job.output_locations)
        self.assertTrue(all(isinstance(value, str) for value in job.output_locations.values()))
        (job_directory / "layout-result.json").write_text(json.dumps({"CACHE_FILENAME": "cache.h5"}))
        with self.assertRaisesRegex(Exception, "TARGET_CACHE_PATH"):
            PipelineJobManager._read_layout_result(job)


class NetworkStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temporary.name)
        self.headers = ["A", "B", "C", "D", "E"]
        # Scores: A-B 9, C-D 8, B-C 5, A-E 1; the other six pairs are absent.
        write_alignment_network(self.path / "net.h5", self.headers,
                                [(0, 1, 9.0), (2, 3, 8.0), (1, 2, 5.0), (0, 4, 1.0)])

    def tearDown(self):
        self.temporary.cleanup()

    def summarize(self, **kwargs):
        from mcp_server.pipeline.Network_Statistics import summarize_network
        return summarize_network(str(self.path / "net.h5"), **kwargs)

    def test_rows_match_the_layout_filters_and_connectivity(self):
        report = self.summarize(thresholds=[5.0], top_edge_percents=[20.0])
        self.assertEqual(report["pairs"], {"stored": 4, "possible": 10, "stored_percent_of_possible": 40.0})
        rows = {(row["filter"], row["value"]): row for row in report["edge_filters"]}
        threshold = rows[("similarity_threshold", 5.0)]
        self.assertEqual(threshold["kept_edges"], 3)
        self.assertEqual(threshold["connectivity"], {"clusters": 1, "isolated_nodes": 1, "largest_cluster_nodes": 4})
        top = rows[("top_edge_percent", 20.0)]  # 20% of 10 possible pairs: the 2 best scores
        self.assertEqual((top["similarity_threshold"], top["kept_edges"]), (8.0, 2))
        self.assertEqual(top["connectivity"], {"clusters": 2, "isolated_nodes": 1, "largest_cluster_nodes": 2})
        self.assertEqual(report["score_summary"]["max"], 9.0)
        self.assertEqual(sum(report["histogram"]["counts"]), 4)

    def test_cutoffs_agree_with_prepare_network(self):
        from desktop.Viewer_State import prepare_network
        report = self.summarize(top_edge_percents=[30.0])
        row = report["edge_filters"][0]
        settings = SimpleNamespace(ALIGNMENT_SCORE="global", NORM_MODE="alignment_length",
                                   SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=30.0, UMAP_MODE=False,
                                   NODE_FASTA_FILE="")
        with h5py.File(self.path / "net.h5", "r") as data:
            _headers, edges, _scores = prepare_network(data, settings=settings, selected_fasta_headers=self.headers)
        self.assertEqual(row["similarity_threshold"], settings.SIMILARITY_THRESHOLD)
        self.assertEqual(row["kept_edges"], len(edges))

    def test_fasta_subsets_report_missing_records(self):
        write_fasta(self.path / "subset.fasta", ["A", "B", "Z"])
        report = self.summarize(fasta_path=str(self.path / "subset.fasta"))
        self.assertEqual(report["nodes"], {"in_network": 5, "kept_for_layout": 2, "fasta_records": 3,
                                           "fasta_records_missing_from_network": 1})
        self.assertEqual(report["pairs"]["stored"], 1)

    def test_invalid_combinations_and_missing_files_are_errors(self):
        from mcp_server.pipeline.Network_Statistics import NetworkStatisticsError
        with self.assertRaisesRegex(NetworkStatisticsError, "unavailable for local"):
            self.summarize(alignment_score="local")
        with self.assertRaisesRegex(NetworkStatisticsError, "not found"):
            self.summarize(fasta_path=str(self.path / "missing.fasta"))

    def test_helper_process_returns_the_report_or_the_error(self):
        from mcp_server.pipeline.Network_Statistics import NetworkStatisticsError, network_statistics
        report = network_statistics({"network_path": str(self.path / "net.h5"), "thresholds": [5.0]})
        self.assertEqual(report["network_type"], "alignment")
        self.assertEqual(report["input_hdf5"], str(self.path / "net.h5"))
        with self.assertRaisesRegex(NetworkStatisticsError, "was not found"):
            network_statistics({"network_path": str(self.path / "absent.h5")})


class LayoutContractTests(unittest.TestCase):
    def test_contract_covers_every_layout_field_and_matches_engine_defaults(self):
        from desktop.Viewer_State import sections
        from Layout_Cache_Generator import LayoutGenerationSettings
        from mcp_server.pipeline.Pipeline_Operations import get_layout_schema, layout_defaults

        schema = get_layout_schema()
        keys = {key for fields in sections("layout").values() for key in fields}
        self.assertEqual(set(schema["fields"]), keys)
        for key, field in schema["fields"].items():
            self.assertTrue(field["description"], key)
        with tempfile.TemporaryDirectory() as temp:
            namespace = SimpleNamespace(SAVED_LAYOUT_DIR=temp, NODE_FASTA_FILE="a.fasta",
                                        INPUT_HDF5="b.h5", SIMILARITY_THRESHOLD=1.0)
            engine = LayoutGenerationSettings.from_namespace(namespace, cache_filename="version_00.h5",
                                                             project_root=temp)
        for key, value in layout_defaults().items():
            if key != "SIMILARITY_THRESHOLD":
                self.assertEqual(getattr(engine, key), value, key)

    def test_individual_arguments_fall_back_to_the_documented_defaults(self):
        from mcp.server.mcpserver.exceptions import ToolError
        from desktop.Viewer_State import decode_document
        from mcp_server.pipeline import Pipeline_Operations

        captured = {}

        def capture(document, project_root):
            captured.update(document)
            raise ValueError("captured")

        with tempfile.TemporaryDirectory() as temp, mock.patch(
                "Layout_Cache_Generator.LayoutGenerationSettings.from_document", side_effect=capture):
            with self.assertRaisesRegex(ToolError, "captured"):
                asyncio.run(Pipeline_Operations.start_layout_job(
                    None, node_fasta_file="a.fasta", input_hdf5="b.h5", directories={"SAVED_LAYOUT_DIR": temp}))
        values = decode_document(captured, "layout")
        for key, value in Pipeline_Operations.layout_defaults().items():
            self.assertEqual(values[key], value, key)


class LayoutExportTests(unittest.TestCase):
    def setUp(self):
        from tests.layout_fixtures import write_inputs
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        write_inputs(self.root)
        (self.root / "viewer_settings.json").write_text(json.dumps({"AUTO_DT": True, "SPRING_K": 7.5}))

    def tearDown(self):
        self.temporary.cleanup()

    def test_partial_overlays_keep_saved_optional_values(self):
        # An overlay without AUTO_DT used to reset the saved true to the default false.
        from utilities.Headless_Settings import export_config_settings
        overlay = {"schema_version": 2, "kind": "layout",
                   "inputs": {"NODE_FASTA_FILE": str(self.root / "set.fasta"),
                              "INPUT_HDF5": str(self.root / "network.h5")},
                   "network": {"SIMILARITY_THRESHOLD": 0.5}}
        (self.root / "overlay.json").write_text(json.dumps(overlay))
        for kwargs in ({"settings_path": str(self.root / "overlay.json")}, {"overlay": overlay}):
            with self.subTest(source=next(iter(kwargs))):
                output = self.root / f"export-{next(iter(kwargs))}.json"
                document = export_config_settings("layout", self.root, str(output), **kwargs)["settings_document"]
                self.assertTrue(document["simulation"]["AUTO_DT"])
                self.assertEqual(document["physics"]["SPRING_K"], 7.5)
                self.assertEqual(document["network"]["SIMILARITY_THRESHOLD"], 0.5)
        explicit = {**overlay, "simulation": {"AUTO_DT": False}}
        document = export_config_settings("layout", self.root, str(self.root / "explicit.json"),
                                          overlay=explicit)["settings_document"]
        self.assertFalse(document["simulation"]["AUTO_DT"])

    def test_mcp_export_takes_inputs_and_fields_and_explains_missing_inputs(self):
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp_server.pipeline import Pipeline_Operations

        async def export(**arguments):
            return await Pipeline_Operations.export_layout_settings(**arguments)

        with mock.patch.object(Pipeline_Operations, "_PROJECT_ROOT", str(self.root)):
            result = asyncio.run(export(node_fasta_file=str(self.root / "set.fasta"),
                                        input_hdf5=str(self.root / "network.h5"),
                                        parameters={"top_edge_percent": 50, "dt": 0.002},
                                        output_path=str(self.root / "out.json")))
            document = result["settings_document"]
            self.assertEqual(document["network"]["TOP_EDGE_PERCENT"], 50)
            self.assertEqual(document["simulation"]["DT"], 0.002)
            self.assertTrue(document["simulation"]["AUTO_DT"])
            self.assertEqual(pathlib.Path(document["inputs"]["INPUT_HDF5"]), self.root / "network.h5")
            with self.assertRaisesRegex(ToolError, "pass node_fasta_file and input_hdf5"):
                asyncio.run(export(output_path=str(self.root / "none.json")))
            with self.assertRaisesRegex(ToolError, "use the node_fasta_file argument"):
                asyncio.run(export(parameters={"NODE_FASTA_FILE": "x.fasta"}))


class ArgumentHintTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_arguments_name_the_accepted_ones_and_the_parameters_object(self):
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp_server.core.Workflow_Dispatch import dispatch
        with self.assertRaises(ToolError) as caught:
            await dispatch("emapssn_pipeline", "export_layout_settings",
                           {"NODE_FASTA_FILE": "a.fasta", "TOP_EDGE_PERCENT": 5}, None)
        message = str(caught.exception)
        self.assertIn("Accepted arguments: node_fasta_file, input_hdf5, parameters", message)
        self.assertIn("use node_fasta_file for NODE_FASTA_FILE", message)
        self.assertIn("Put settings fields such as TOP_EDGE_PERCENT inside the parameters object", message)
        with self.assertRaises(ToolError) as caught:
            await dispatch("emapssn_pipeline", "get_job", {"job_id": "x", "jobid": "y"}, None)
        self.assertIn("Accepted arguments: job_id", str(caught.exception))
        self.assertNotIn("parameters object", str(caught.exception))


class InspectionAdditionsTests(unittest.TestCase):
    def test_fasta_lengths_and_network_scoring_provenance(self):
        from mcp_server.pipeline.Pipeline_File_Inspection import inspect_local
        with tempfile.TemporaryDirectory() as temp:
            fasta = pathlib.Path(temp) / "set.fasta"
            fasta.write_text(">a\nACDE\n>b\nAC\nDEFG\n>c\nA\n>d\nacde\n", encoding="utf-8")
            report = inspect_local(fasta)
            self.assertEqual(report["metadata"]["sequence_lengths"],
                             {"min": 1, "p5": 1, "median": 4, "p95": 6, "max": 6})
            self.assertEqual(report["metadata"]["duplicate_sequences"], 1)
            self.assertIn("masks other invalid characters as X", report["findings"][0]["message"])
            self.assertEqual(report["metadata"]["header_terms"], {})
            self.assertEqual(report["metadata"]["header_examples"], ["a", "b", "c"])
            annotated = pathlib.Path(temp) / "annotated.fasta"
            annotated.write_text(">x1 Partial protein, partial\nAC\n>x2 fragment\nAD\n>x3 hypothetical\nAE\n",
                                 encoding="utf-8")
            terms = inspect_local(annotated)["metadata"]["header_terms"]
            # Counted per header and exact spelling: the sanitize filter is case-sensitive.
            self.assertEqual(terms, {"Partial": 1, "fragment": 1, "hypothetical": 1, "partial": 1})
            network = pathlib.Path(temp) / "net.h5"
            write_alignment_network(network, ["A", "B"], [(0, 1, 2.0)])
            with h5py.File(network, "a") as hf:
                hf.attrs["gap_penalties"] = np.asarray([-2.0, -0.5], dtype=np.float32)
                hf.attrs["matmul_precision"] = "ieee_fp32"
            metadata = inspect_local(network)["metadata"]
        self.assertEqual(metadata["gap_penalties"], {"local": -2.0, "global": -0.5})
        self.assertEqual(metadata["matmul_precision"], "ieee_fp32")


class LayoutSummaryTests(unittest.TestCase):
    def test_summary_counts_nodes_edges_clusters_and_missing_records(self):
        from utilities.Headless_Settings import layout_summary
        generated = SimpleNamespace(
            full_headers=["A x", "B", "C", "D"],
            edges=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
            edge_scores=np.asarray([3.0, 2.5], dtype=np.float32),
            fasta_records=[("A x", "AC"), ("B", "AC"), ("C", "AC"), ("D", "AC"), ("Z", "AC")],
            effective_similarity_threshold=2.5,
        )
        settings = SimpleNamespace(UMAP_MODE=False, TOP_EDGE_PERCENT=None, SIMILARITY_THRESHOLD=2.5,
                                   UMAP_NEIGHBORS=15, LAYOUT_DIMENSIONS=2)
        summary = layout_summary(generated, settings)
        self.assertEqual(summary, {
            "layout_mode": "physics", "layout_dimensions": 2, "nodes": 4, "fasta_records": 5,
            "fasta_records_missing_from_network": 1, "edges": 2,
            "edge_filter": {"mode": "similarity_threshold", "value": 2.5},
            "effective_similarity_threshold": 2.5, "kept_score_range": [2.5, 3.0],
            "clusters": 1, "isolated_nodes": 1, "largest_cluster_nodes": 3,
        })


if __name__ == "__main__":
    unittest.main()
