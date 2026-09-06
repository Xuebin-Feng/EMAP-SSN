import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mcp_server import Pipeline_File_Inspection as inspection
from utilities.Embedding_HDF5 import create_metadata_first_file


class FileInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "selected.data"

    def tearDown(self):
        self.temp.cleanup()

    def inspect(self, kind="auto", **kwargs):
        return inspection.inspect_local(self.path, kind, project_root=ROOT, **kwargs)

    def embedding(self, complete=True, missing=False):
        with h5py.File(self.path, "w") as hf:
            create_metadata_first_file(hf, ["a", "b"], ["AC", "ACD"], "esmc_300m", "float32")
            hf["embeddings"].create_dataset("a", shape=(2, 4), dtype="float32")
            if not missing: hf["embeddings"].create_dataset("b", shape=(3, 4), dtype="float32")
            hf.attrs["generation_complete"] = complete

    def network(self, edges=1, alignment=False):
        with h5py.File(self.path, "w") as hf:
            hf.attrs["model_name"] = "esmc_300m" if alignment else "BLAST"
            hf.create_dataset("headers", data=["a", "b", "c"], dtype=h5py.string_dtype())
            for key in ("i", "j"): hf.create_dataset(key, shape=(edges,), dtype="uint32")
            if alignment:
                hf.create_dataset("seq_lens", data=[2, 3, 4], dtype="uint32")
                for key in ("l_score", "g_score"): hf.create_dataset(key, shape=(edges,), dtype="float32")
                for key in ("l_len", "g_len"): hf.create_dataset(key, shape=(edges,), dtype="uint32")
            else: hf.create_dataset("score", shape=(edges,), dtype="float32")

    def sparse(self):
        with h5py.File(self.path, "w") as hf:
            hf.attrs["shape"] = [2, 3]
            group = hf.create_group("matrix")
            group.attrs["shape"] = [2, 3]
            group.create_dataset("data", shape=(2,), dtype="uint8")
            group.create_dataset("indices", shape=(2,), dtype="int32")
            group.create_dataset("indptr", shape=(3,), dtype="int32")
            hf.create_dataset("headers", data=["a", "b"], dtype=h5py.string_dtype())
            for key, value in {"header_map": {"a": 0, "b": 1}, "aa_map": {"A": 1}, "int_to_aa": {"1": "A"}}.items():
                hf.create_dataset(key, data=json.dumps(value))

    def test_embedding_completion_and_contradictions(self):
        self.embedding()
        result = self.inspect()
        self.assertEqual(result["detected_format"], "embedding")
        self.assertEqual(result["structural_validity"], "valid")
        self.assertEqual(result["generation_completion"]["status"], "complete")
        self.embedding(False, True)
        result = self.inspect()
        self.assertEqual(result["generation_completion"]["status"], "incomplete")
        self.assertEqual(result["structural_validity"], "valid")
        self.embedding(True, True)
        result = self.inspect()
        self.assertEqual(result["structural_validity"], "invalid")
        self.assertEqual(result["generation_completion"]["status"], "unknown")
        self.embedding()
        with h5py.File(self.path, "r+") as hf:
            del hf["embeddings/b"]
            hf["embeddings"].create_dataset("b", shape=(7, 4), dtype="float32")
        self.assertEqual(self.inspect()["structural_validity"], "invalid")

    def test_network_sparse_is_not_unfinished(self):
        self.network()
        result = self.inspect()
        self.assertEqual(result["structural_validity"], "valid")
        self.assertEqual(result["generation_completion"]["status"], "unknown")
        self.assertEqual(result["network_pair_coverage"]["status"], "sparse")
        self.assertEqual(result["network_pair_coverage"]["expected"], 3)
        self.network(3)
        result = self.inspect()
        self.assertEqual(result["network_pair_coverage"]["status"], "all-pairs count reached")
        self.assertFalse(result["network_pair_coverage"]["unique_pairs_verified"])
        self.assertEqual(result["generation_completion"]["status"], "unknown")
        self.network(1, True)
        with h5py.File(self.path, "r+") as hf: hf.attrs["sparsity_keep_count"] = 2
        self.assertEqual(self.inspect()["generation_completion"]["status"], "unknown")
        self.assertEqual(self.inspect()["structural_validity"], "valid")
        with h5py.File(self.path, "r+") as hf:
            hf.attrs["complete"] = True
            hf.create_group("_resume")
        result = self.inspect()
        self.assertEqual(result["structural_validity"], "invalid")
        self.assertNotEqual(result["generation_completion"]["status"], "complete")
        self.network(1, True)
        with h5py.File(self.path, "r+") as hf: hf.create_group("_resume")
        self.assertEqual(self.inspect()["generation_completion"]["status"], "incomplete")

    def test_network_bad_shapes_missing_objects_and_counts(self):
        for change in ("missing", "rank", "count", "dtype"):
            with self.subTest(change=change):
                self.network()
                with h5py.File(self.path, "r+") as hf:
                    del hf["score"]
                    if change == "rank": hf.create_dataset("score", shape=(1, 1), dtype="float32")
                    if change == "count": hf.create_dataset("score", shape=(2,), dtype="float32")
                    if change == "dtype": hf.create_dataset("score", shape=(1,), dtype="int32")
                self.assertEqual(self.inspect()["structural_validity"], "invalid")
        self.network(4)
        self.assertEqual(self.inspect()["structural_validity"], "invalid")

    def test_sparse_msa_structure_and_mappings(self):
        self.sparse()
        result = self.inspect()
        self.assertEqual(result["detected_format"], "sparse_msa")
        self.assertEqual(result["structural_validity"], "valid")
        self.assertEqual(result["generation_completion"]["status"], "unknown")
        with h5py.File(self.path, "r+") as hf: hf["matrix"].attrs["shape"] = [3, 3]
        self.assertEqual(self.inspect()["structural_validity"], "invalid")
        self.sparse()
        with h5py.File(self.path, "r+") as hf:
            del hf["header_map"]
            hf.create_dataset("header_map", data=json.dumps({"a": 9}))
        self.assertEqual(self.inspect()["structural_validity"], "invalid")

    def test_numeric_payloads_are_never_read_and_sources_unchanged(self):
        original = h5py.Dataset.__getitem__
        def guarded(dataset, key, *args, **kwargs):
            if dataset.name.startswith(("/embeddings/", "/matrix/")) or dataset.name in ("/i", "/j", "/score", "/l_score", "/g_score", "/l_len", "/g_len", "/seq_lens"):
                self.fail("Numerical payload read: " + dataset.name)
            return original(dataset, key, *args, **kwargs)
        for fixture in (self.embedding, self.network, self.sparse):
            fixture()
            before = self.path.read_bytes()
            with mock.patch.object(h5py.Dataset, "__getitem__", guarded):
                self.assertEqual(self.inspect()["structural_validity"], "valid")
            self.assertEqual(before, self.path.read_bytes())
            self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_fasta_alignment_and_raw_sanitization_warnings(self):
        self.path.write_text(">a\nac?\n>b\nACD\n")
        result = self.inspect()
        self.assertEqual(result["structural_validity"], "valid")
        self.assertEqual(result["generation_completion"]["status"], "unknown")
        self.assertEqual(result["metadata"]["sequence_count"], 2)
        self.assertTrue(result["findings"])
        self.path.write_text(">a\nAC\n>b\nACD\n")
        self.assertEqual(self.inspect("alignment_fasta")["structural_validity"], "invalid")
        self.assertEqual(self.inspect(tool_id="sparse_msa_converter")["detected_format"], "alignment_fasta")
        self.path.write_text("ACD\n>a\nAC\n")
        self.assertEqual(self.inspect("fasta")["structural_validity"], "invalid")
        self.path.write_text("")
        self.assertEqual(self.inspect("fasta")["structural_validity"], "invalid")

    def test_blast_layouts_columns_and_no_numeric_scan(self):
        self.path.write_text("q\ts\t" + "\t".join(["notnumeric"] * 10) + "\n")
        self.assertEqual(self.inspect()["structural_validity"], "unknown")
        self.assertEqual(self.inspect("blast_tabular")["structural_validity"], "valid")
        self.path.write_text("q\ts\te\n")
        self.assertEqual(self.inspect("blast_tabular")["structural_validity"], "invalid")
        custom = {"BLAST_LAYOUT": "custom_columns", "QUERY_COLUMN": 1, "SUBJECT_COLUMN": 2, "EVALUE_COLUMN": 3}
        self.assertEqual(self.inspect("blast_tabular", parameters=custom)["structural_validity"], "valid")
        self.path.write_text("q\t\te\n")
        self.assertEqual(self.inspect("blast_tabular", parameters=custom)["structural_validity"], "invalid")
        self.path.write_text("# BLASTP\n# Query: q\n# Fields: subject id, evalue\ns\tunknown\n")
        self.assertEqual(self.inspect(parameters={"BLAST_LAYOUT": "outfmt7_fields"})["structural_validity"], "valid")
        self.path.write_text("# Query: q\ns\t0\n")
        self.assertEqual(self.inspect("blast_tabular", parameters={"BLAST_LAYOUT": "outfmt7_fields"})["structural_validity"], "invalid")

    def test_settings_sections_and_strict_validation(self):
        self.path.write_text(json.dumps({"DIRECTORIES": {}, "Sanitize_Sequences.py": {"INPUT_FASTA": "future.fasta"}}))
        self.assertEqual(self.inspect()["structural_validity"], "valid")
        self.assertEqual(self.inspect(tool_id="sanitize_sequences")["structural_validity"], "valid")
        self.path.write_text(json.dumps({"DIRECTORIES": {}, "Sanitize_Sequences.py": {"OVER_WRITE": "true"}}))
        self.assertEqual(self.inspect(tool_id="sanitize_sequences")["structural_validity"], "invalid")
        self.path.write_text('{"DIRECTORIES": {}, "Typo.py": {}}')
        self.assertEqual(self.inspect()["structural_validity"], "invalid")
        self.path.write_text('{"DIRECTORIES":')
        self.assertEqual(self.inspect()["structural_validity"], "invalid")

    def test_limits_unknown_formats_and_read_failures(self):
        self.path.write_text("unrecognized text")
        self.assertEqual(self.inspect()["structural_validity"], "unknown")
        self.assertEqual(self.inspect("network")["structural_validity"], "unknown")
        self.path.write_text(">a\nAC\n")
        result = self.inspect(budget=-1)
        self.assertFalse(result["inspection_finished"])
        self.assertEqual(result["generation_completion"]["status"], "unknown")
        with mock.patch.object(inspection, "MAX_TEXT_LINE", 1):
            self.assertFalse(self.inspect()["inspection_finished"])
        self.path.write_text("".join(">\n\n" for _ in range(100)))
        result = self.inspect()
        self.assertEqual(len(result["findings"]), inspection.MAX_FINDINGS)
        self.assertGreater(result["findings_omitted"], 0)
        self.path.unlink()
        self.assertEqual(self.inspect()["structural_validity"], "unknown")
        self.path.write_bytes(b"\x89HDF\r\n\x1a\ntruncated")
        self.assertEqual(self.inspect()["generation_completion"]["status"], "unknown")
        with h5py.File(self.path, "w") as hf: hf.create_dataset("positions", shape=(1, 3), dtype="float32")
        self.assertEqual(self.inspect()["structural_validity"], "unknown")

    def test_file_changes_invalidate_completion(self):
        self.embedding()
        original = inspection.Inspector.embedding
        def changed(inspector, hf):
            original(inspector, hf)
            stat = inspector.path.stat()
            os.utime(inspector.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000))
        with mock.patch.object(inspection.Inspector, "embedding", changed):
            result = self.inspect()
        self.assertTrue(result["unstable"])
        self.assertEqual(result["structural_validity"], "unknown")
        self.assertEqual(result["generation_completion"]["status"], "unknown")

    def test_helper_relative_absolute_context_and_invalid_requests(self):
        self.path.write_text(">a\nAC\n")
        relative = inspection.inspect_pipeline_file(self.path.name, self.root)
        absolute = inspection.inspect_pipeline_file(str(self.path), ROOT)
        self.assertEqual(relative["path"], absolute["path"])
        self.assertEqual(relative["structural_validity"], "valid")
        for kwargs in ({"tool_id": "bogus"}, {"parameters": {}},
                       {"tool_id": "parse_blast_output", "parameters": {"QUERY_COLUMN": "1"}},
                       {"tool_id": "parse_blast_output", "parameters": {"TYPO": 1}}):
            with self.subTest(kwargs=kwargs), mock.patch.object(inspection.subprocess, "run") as run:
                with self.assertRaises((KeyError, ValueError)): inspection.inspect_pipeline_file(self.path, ROOT, **kwargs)
                run.assert_not_called()

    def test_timeout_and_malformed_helper_reports(self):
        self.path.write_text(">a\nAC\n")
        with mock.patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("helper", 20)):
            result = inspection.inspect_pipeline_file(self.path, ROOT)
            self.assertEqual(result["structural_validity"], "unknown")
            self.assertFalse(result["inspection_finished"])
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0, stdout="{}")):
            self.assertEqual(inspection.inspect_pipeline_file(self.path, ROOT)["structural_validity"], "unknown")

    def test_incomplete_flags_metadata_limits_and_unreadable_inputs(self):
        for fixture in (self.embedding, self.network, self.sparse):
            fixture()
            with h5py.File(self.path, "r+") as hf:
                flag = "generation_complete" if fixture == self.embedding else "complete"
                hf.attrs[flag] = False
            self.assertEqual(self.inspect()["generation_completion"]["status"], "incomplete")
        self.embedding()
        with mock.patch.object(inspection, "MAX_METADATA_RECORDS", 0):
            result = self.inspect()
        self.assertFalse(result["inspection_finished"])
        self.assertEqual(result["generation_completion"]["status"], "unknown")
        with mock.patch.object(Path, "stat", side_effect=PermissionError("access denied")):
            result = self.inspect()
        self.assertEqual(result["structural_validity"], "unknown")
        self.assertIn("access denied", str(result["findings"]))
        for kind in ("embedding", "network", "sparse_msa"):
            self.path.write_bytes(b"\x89HDF\r\n\x1a\ntruncated")
            result = self.inspect(kind)
            self.assertFalse(result["inspection_finished"])
            self.assertEqual(result["generation_completion"]["status"], "unknown")

    def test_inspection_timeout_reaps_only_its_helper(self):
        real_run, real_popen = subprocess.run, subprocess.Popen
        children = []
        def popen(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child
        def run(*args, **kwargs):
            return real_run([sys.executable, "-c", "import time;time.sleep(5)"],
                            capture_output=True, timeout=0.1)
        with mock.patch.object(subprocess, "Popen", side_effect=popen), \
                mock.patch.object(subprocess, "run", side_effect=run):
            result = inspection.inspect_pipeline_file(self.path, ROOT)
        self.assertFalse(result["inspection_finished"])
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())


if __name__ == "__main__":
    unittest.main()
