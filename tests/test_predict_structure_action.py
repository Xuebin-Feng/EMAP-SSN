# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""emapssn_pipeline(action="predict_structure"): the call, its job and the records that come back.

No model runs: the job tests use a stand-in Predict_Structures.py in a stand-in project, as the
benchmark's job tests do, and the call tests replace the job manager.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcp.server.mcpserver.exceptions import ToolError

from mcp_server.core.Workflow_Dispatch import REGISTRY, dispatch
from mcp_server.pipeline import Pipeline_Operations as operations
from mcp_server.pipeline.Pipeline_Jobs import PipelineJobManager
from mcp_server.pipeline.Pipeline_Settings import normalize_pipeline_settings

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"

# Stands in for src/tools/Predict_Structures.py: reads the settings snapshot it is given, writes one PDB for
# each behaviour entry into STRUCTURE_DIR, and the records to the file the job names, as the real tool does.
STAND_IN_TOOL = r'''
import json, os, sys, time
document = json.load(open(sys.argv[1], encoding="utf-8"))
folder = document["DIRECTORIES"]["STRUCTURE_DIR"]
behaviour = json.load(open(os.path.join(os.path.dirname(__file__), "behaviour.json"), encoding="utf-8"))
os.makedirs(folder, exist_ok=True)
records = []
for name in behaviour["names"]:
    path = os.path.join(folder, name + ".pdb")
    open(path, "w").write("ATOM\n")
    records.append({"name": name, "path": path, "length": 33, "status": "folded", "mean_plddt": 80.5, "note": ""})
variable = os.environ.get("SSN_TOOL_RESULT_FILE")
if variable and behaviour.get("write_result", True):
    json.dump({"model": document["Predict_Structures.py"]["PREDICTION_MODEL"], "structures": records},
              open(variable, "w", encoding="utf-8"))
open(os.path.join(os.path.dirname(__file__), "seen_variable.txt"), "w").write(variable or "none")
time.sleep(behaviour.get("sleep", 0))
raise SystemExit(behaviour["exit"])
'''


class JobTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = Path(self.enterContext(tempfile.TemporaryDirectory(ignore_cleanup_errors=True)))
        self.project = self.temporary / "project"
        script = self.project / "src" / "tools" / "Predict_Structures.py"
        script.parent.mkdir(parents=True)
        script.write_text(STAND_IN_TOOL, encoding="utf-8")
        self.behaviour = script.with_name("behaviour.json")
        self.folder = self.temporary / "Folded"
        self.manager = PipelineJobManager(self.project, max_pending=2, history_limit=4, termination_grace=0.5,
                                          temporary_parent=self.temporary)
        await self.manager.start()
        self.addAsyncCleanup(self.manager.close)

    def document(self, **parameters):
        parameters.setdefault("SEQUENCE", SEQUENCE)
        preview = normalize_pipeline_settings("predict_structures", str(self.project), parameters=parameters,
                                              directories={"STRUCTURE_DIR": str(self.folder)})
        self.assertTrue(preview["valid"], preview["errors"])
        return preview["settings_document"]

    async def run_job(self, behaviour, collect=True, **parameters):
        self.behaviour.write_text(json.dumps(behaviour), encoding="utf-8")
        submitted = await self.manager.submit("predict_structures", self.document(**parameters), collect_structures=collect)
        self.assertEqual(submitted["tool_id"], "predict_structures")
        return await self.manager.wait_for_terminal(submitted["job_id"])

    async def test_a_finished_job_hands_back_the_records_and_lists_only_the_pdb_files(self):
        job = await self.run_job({"names": ["a", "b"], "exit": 0}, PREDICTION_MODEL="local")
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual([record["name"] for record in job["result"]["structures"]], ["a", "b"])
        self.assertEqual(job["result"]["structures"][0]["mean_plddt"], 80.5)
        self.assertEqual(sorted(Path(item["path"]).name for item in job["output_files"]), ["a.pdb", "b.pdb"])
        # The records file is in the job's own folder, never in the structures folder.
        self.assertEqual(sorted(path.name for path in self.folder.iterdir()), ["a.pdb", "b.pdb"])
        self.assertEqual(Path(job["settings_snapshot"]).parent, Path(job["stdout_log"]).parent)
        self.assertTrue((Path(job["stdout_log"]).parent / "predict-structures-result.json").is_file())

    async def test_a_failed_run_keeps_the_records_it_made(self):
        job = await self.run_job({"names": ["a"], "exit": 1})
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["result"]["structures"][0]["name"], "a")

    async def test_a_missing_result_after_success_fails_the_job(self):
        job = await self.run_job({"names": ["a"], "exit": 0, "write_result": False})
        self.assertEqual(job["status"], "failed")
        self.assertIn("structure prediction process exited with code 0, but its result could not be read",
                      job["failure_message"])

    async def test_a_job_started_without_the_action_is_not_told_where_to_write_records(self):
        job = await self.run_job({"names": ["a"], "exit": 0}, collect=False)
        self.assertEqual(job["status"], "succeeded")
        self.assertIsNone(job["result"])
        self.assertEqual((self.project / "src" / "tools" / "seen_variable.txt").read_text(), "none")

    async def test_only_that_tool_takes_a_result_file(self):
        with self.assertRaisesRegex(Exception, "Only predict_structures"):
            await self.manager.submit("sanitize_sequences", {"DIRECTORIES": {}, "Sanitize_Sequences.py": {}},
                                      collect_structures=True)


BASE_PAYLOAD = {
    "job_id": "job-1", "tool_id": "predict_structures", "status": "queued", "queue_position": 1,
    "created_at": "now", "started_at": None, "finished_at": None, "exit_code": None, "failure_message": None,
    "cancellation_requested": False, "settings_snapshot": "snapshot.json", "stdout_log": "out", "stderr_log": "err",
    "output_locations": {}, "output_files": [], "output_files_omitted": 0, "result": None,
}
RECORD = {"name": "my_protein", "path": "my_protein.pdb", "length": 33, "status": "folded",
          "mean_plddt": 80.5, "note": ""}


class ActionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.jobs = mock.Mock()
        self.jobs.submit = mock.AsyncMock(return_value=dict(BASE_PAYLOAD))
        self.jobs.wait = mock.AsyncMock(return_value=dict(BASE_PAYLOAD, status="running"))
        self.enterContext(mock.patch.object(operations, "_context", return_value=mock.Mock(jobs=self.jobs)))

    async def call(self, **arguments):
        return await dispatch("emapssn_pipeline", "predict_structure", arguments, mock.Mock())

    def section(self):
        document = self.jobs.submit.await_args.args[1]
        return document["Predict_Structures.py"], document

    def test_the_action_is_listed_described_and_its_example_validates(self):
        self.assertIn("predict_structure", REGISTRY["emapssn_pipeline"])
        action = REGISTRY["emapssn_pipeline"]["predict_structure"]
        action.model.model_validate(action.example)
        self.assertEqual(set(action.model.model_fields),
                         {"sequence", "name", "fasta_file", "model", "max_sequences", "output_dir", "wait_seconds"})
        fields = action.model.model_fields
        self.assertEqual((fields["model"].default, fields["max_sequences"].default,
                          fields["wait_seconds"].default), ("local", 20, 50.0))
        self.assertNotIn("device", fields)

    async def test_help_and_describe_show_it(self):
        listing = await dispatch("emapssn_pipeline", "help", {}, None)
        self.assertIn("predict_structure", [entry["action"] for entry in listing["actions"]])
        described = await dispatch("emapssn_pipeline", "describe", {"action": "predict_structure"}, None)
        self.assertEqual(described["example"]["action"], "predict_structure")
        self.assertIn("sequence", described["arguments_schema"]["properties"])

    async def test_a_sequence_with_a_name_becomes_the_typed_input_of_the_tool(self):
        await self.call(sequence=SEQUENCE, name="my protein")
        section, document = self.section()
        self.assertEqual((section["BATCH_INPUT"], section["SEQUENCE"], section["SEQUENCE_NAME"]), (False, SEQUENCE, "my protein"))
        self.assertEqual((section["PREDICTION_MODEL"], section["MAX_SEQUENCES"]), ("local", 20))
        self.assertNotIn("DEVICE_SELECTION", section)
        self.assertEqual(Path(document["DIRECTORIES"]["STRUCTURE_DIR"]),
                         PROJECT_ROOT / "Analysis_Results" / "Folded_Structure")
        self.assertEqual(self.jobs.submit.await_args.kwargs, {"collect_structures": True})
        self.assertEqual(self.jobs.submit.await_args.args[0], "predict_structures")

    async def test_a_fasta_becomes_the_batch_input_and_an_output_folder_is_passed_on(self):
        await self.call(fasta_file="/data/set.fasta", output_dir=str(Path(tempfile.gettempdir()) / "out"),
                        max_sequences=3)
        section, document = self.section()
        self.assertEqual((section["BATCH_INPUT"], section["INPUT_FASTA"], section["MAX_SEQUENCES"]),
                         (True, "/data/set.fasta", 3))
        self.assertEqual(Path(document["DIRECTORIES"]["STRUCTURE_DIR"]), Path(tempfile.gettempdir()) / "out")

    async def test_remote_is_set_only_when_the_call_says_so(self):
        reply = await self.call(sequence=SEQUENCE)
        self.assertEqual(self.section()[0]["PREDICTION_MODEL"], "local")
        self.assertIsNone(reply.get("notice"))
        reply = await self.call(sequence=SEQUENCE, model="remote")
        self.assertEqual(self.section()[0]["PREDICTION_MODEL"], "remote")
        self.assertIn("sent to the Biohub API", reply["notice"])

    async def test_arguments_that_make_no_sense_are_refused_before_anything_is_queued(self):
        for arguments, pattern in (
            ({}, "exactly one of sequence and fasta_file"),
            ({"sequence": SEQUENCE, "fasta_file": "set.fasta"}, "exactly one of sequence and fasta_file"),
            ({"sequence": "  ", "fasta_file": ""}, "exactly one of sequence and fasta_file"),
            ({"fasta_file": "set.fasta", "name": "x"}, "name applies to a sequence only"),
            ({"sequence": "1234"}, "empty after sanitization"),
            ({"sequence": SEQUENCE, "device": "cpu"}, "device"),
            ({"sequence": SEQUENCE, "model": "hosted"}, "model"),
            ({"sequence": SEQUENCE, "max_sequences": 0}, "max_sequences"),
            ({"sequence": SEQUENCE, "wait_seconds": 51}, "wait_seconds"),
            ({"sequence": SEQUENCE, "unknown": 1}, "unknown"),
        ):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ToolError, pattern):
                await self.call(**arguments)
        self.jobs.submit.assert_not_awaited()

    async def test_a_job_that_finishes_within_the_wait_returns_its_records(self):
        self.jobs.wait.return_value = dict(BASE_PAYLOAD, status="succeeded", exit_code=0,
                                           result={"model": "local", "structures": [RECORD]})
        reply = await self.call(sequence=SEQUENCE, wait_seconds=12)
        self.jobs.wait.assert_awaited_once_with("job-1", timeout=12.0)
        self.assertEqual(reply["status"], "succeeded")
        self.assertEqual(reply["structures"], [RECORD])
        self.assertIsNone(reply["next_step"])

    async def test_a_job_still_running_after_the_wait_says_to_continue_with_wait_job(self):
        reply = await self.call(sequence=SEQUENCE)
        self.jobs.wait.assert_awaited_once_with("job-1", timeout=50.0)
        self.assertEqual(reply["status"], "running")
        self.assertIsNone(reply["structures"])
        self.assertIn("wait_job", reply["next_step"])
        self.assertIn("job-1", reply["next_step"])

    async def test_a_failed_job_keeps_its_records_and_failure_message(self):
        record = dict(RECORD, status="failed", mean_plddt=None, note="a structure named x.pdb already exists for a different sequence")
        self.jobs.wait.return_value = dict(BASE_PAYLOAD, status="failed", exit_code=1,
                                           failure_message="The pipeline process exited with code 1.",
                                           result={"structures": [record]})
        reply = await self.call(sequence=SEQUENCE)
        self.assertEqual(reply["structures"][0]["status"], "failed")
        self.assertIn("different sequence", reply["structures"][0]["note"])

    def test_list_tools_has_a_block_for_structure_prediction_beside_layout_and_benchmark(self):
        catalog = operations.list_pipeline_tools()
        block = catalog.structure_prediction
        self.assertEqual((block["actions"], block["job_tool_id"]), (["predict_structure"], "predict_structures"))
        self.assertIn("predict_structure", {tool["action"] for tool in block["examples"]})
        for example in block["examples"]:
            placeholders = {key: (SEQUENCE if key == "sequence" else value) for key, value in example["arguments"].items()}
            self.assertEqual(example["action"], "predict_structure")
            self.assertTrue(set(placeholders) <= set(REGISTRY["emapssn_pipeline"]["predict_structure"].model.model_fields))
        self.assertEqual(catalog.layout["job_tool_id"], "generate_layout_cache")
        self.assertIn("predict_structures", {tool.tool_id for tool in catalog.tools})


if __name__ == "__main__":
    unittest.main()
