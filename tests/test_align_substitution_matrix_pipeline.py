import os
import pathlib
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import h5py
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILITIES_DIR = PROJECT_ROOT / "src" / "utilities"
TOOLS_DIR = PROJECT_ROOT / "src" / "tools"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import Align_Substitution_Matrix as substitution_matrix


class SubstitutionMatrixPipelineTests(unittest.TestCase):
    @staticmethod
    def _metadata(**overrides):
        metadata = {
            "input_fasta": "input.fasta",
            "input_fasta_sha256": "input-checksum",
            "sanitized_manifest_sha256": "manifest-checksum",
            "sequence_count": 4,
            "matrix": "BLOSUM62",
            "num_threads": 2,
            "blastp_version": "blastp: 2.17.0+",
        }
        metadata.update(overrides)
        return metadata

    @staticmethod
    def _blast_line(source, target, evalue):
        return (
            f"{source} {target} 100.0 10 0 0 1 10 1 10 "
            f"{evalue} 50.0\n"
        )

    def test_blast_input_uses_canonical_embedding_sanitization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.fasta")
            output_path = os.path.join(temp_dir, "work", "safe.fasta")
            with open(input_path, "w", encoding="utf-8") as fasta_file:
                fasta_file.write(
                    ">short?\n"
                    "--ac?d*\n"
                    ">a much longer/header\n"
                    "ACXD\n"
                    ">other[one]\n"
                    "***mnp--q***\n"
                )

            headers, sequences = substitution_matrix.prepare_blast_fasta(
                input_path,
                output_path,
            )

            self.assertEqual(
                headers,
                ["a much longer_header", "other(one)"],
            )
            self.assertEqual(sequences, ["ACXD", "MNPXXQ"])
            with open(output_path, "r", encoding="utf-8") as safe_fasta:
                self.assertEqual(
                    safe_fasta.read(),
                    ">0\nACXD\n>1\nMNPXXQ\n",
                )

    def test_temporary_workspace_is_derived_inside_network_directory(self):
        self.assertEqual(
            os.path.normcase(os.path.dirname(substitution_matrix.SAFE_TEMP_DIR)),
            os.path.normcase(os.path.normpath(substitution_matrix.NETWORK_DIR)),
        )
        self.assertTrue(
            os.path.basename(substitution_matrix.SAFE_TEMP_DIR).endswith(
                "_[BLAST]_EValue_temp"
            )
        )

    def test_tools_panel_keeps_fasta_input_but_not_temp_directory_control(self):
        tools_source = (PROJECT_ROOT / "src" / "EMAPSSN_Tools.py").read_text(
            encoding="utf-8"
        )
        combined_start = tools_source.index('"Sequence_Similarity_Calculations"')
        substitution_start = tools_source.index(
            '"Align_Substitution_Matrix.py": [',
            combined_start,
        )
        substitution_end = tools_source.index(
            '"Parse_BLAST_Output.py": [',
            substitution_start,
        )
        substitution_panel = tools_source[substitution_start:substitution_end]

        self.assertIn('"var_name": "INPUT_FASTA"', substitution_panel)
        self.assertNotIn('"var_name": "SAFE_TEMP_DIR"', substitution_panel)
        self.assertNotIn(
            '("SAFE_TEMP_DIR", "BLASTP_DIR")',
            tools_source,
        )

    def test_hdf5_batch_is_bounded_attributed_and_reusable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_dir = os.path.join(temp_dir, "batches")
            os.makedirs(batch_dir)
            result_path = os.path.join(temp_dir, "result_query.txt")
            with open(result_path, "w", encoding="utf-8") as result:
                result.write(self._blast_line(0, 1, "1e-10"))
                result.write(self._blast_line(0, 2, "1e-20"))
                result.write(self._blast_line(1, 3, "1e-30"))
                result.write(self._blast_line(3, 0, "1e-40"))

            flush_sizes = []
            real_append = substitution_matrix._append_edge_buffer

            def recording_append(hf, sources, targets, scores):
                flush_sizes.append(len(sources))
                return real_append(hf, sources, targets, scores)

            with mock.patch.object(
                substitution_matrix,
                "BATCH_DIR",
                batch_dir,
            ), mock.patch.object(
                substitution_matrix,
                "_append_edge_buffer",
                side_effect=recording_append,
            ):
                first = substitution_matrix.parse_result_to_batch(
                    result_path,
                    "query-checksum",
                    self._metadata(),
                    batch_size=2,
                )

            self.assertFalse(first["reused"])
            self.assertEqual(first["edges"], 4)
            self.assertTrue(flush_sizes)
            self.assertLessEqual(max(flush_sizes), 2)

            with h5py.File(first["path"], "r") as batch:
                self.assertEqual(
                    set(batch.attrs),
                    set(substitution_matrix.BATCH_ATTRIBUTE_NAMES),
                )
                self.assertNotIn("blastp_dir", batch.attrs)
                self.assertNotIn("blastp_executable", batch.attrs)
                self.assertNotIn("cache_schema_version", batch.attrs)
                np.testing.assert_array_equal(batch["i"][:], [0, 0, 0, 1])
                np.testing.assert_array_equal(batch["j"][:], [1, 2, 3, 3])

            # BATCH_SIZE is deliberately absent from compatibility metadata.
            with mock.patch.object(substitution_matrix, "BATCH_DIR", batch_dir):
                resumed = substitution_matrix.parse_result_to_batch(
                    result_path,
                    "query-checksum",
                    self._metadata(),
                    batch_size=1,
                )
            self.assertTrue(resumed["reused"])

    def test_reciprocal_hits_are_canonicalized_and_keep_the_best_score(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_dir = os.path.join(temp_dir, "batches")
            os.makedirs(batch_dir)
            result_path = os.path.join(temp_dir, "reciprocal.txt")
            with open(result_path, "w", encoding="utf-8") as result:
                result.write(self._blast_line(0, 1, "1e-10"))
                result.write(self._blast_line(1, 0, "1e-30"))

            with mock.patch.object(substitution_matrix, "BATCH_DIR", batch_dir):
                record = substitution_matrix.parse_result_to_batch(
                    result_path,
                    "reciprocal-query",
                    self._metadata(),
                    batch_size=1,
                )

            self.assertEqual(record["edges"], 1)
            with h5py.File(record["path"], "r") as batch:
                np.testing.assert_array_equal(batch["i"][:], [0])
                np.testing.assert_array_equal(batch["j"][:], [1])
                np.testing.assert_allclose(batch["score"][:], [30.0])

    def test_batch_identity_attribute_changes_invalidate_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_dir = os.path.join(temp_dir, "batches")
            os.makedirs(batch_dir)
            result_path = os.path.join(temp_dir, "result.txt")
            with open(result_path, "w", encoding="utf-8") as result:
                result.write(self._blast_line(0, 1, "1e-10"))

            with mock.patch.object(substitution_matrix, "BATCH_DIR", batch_dir):
                record = substitution_matrix.parse_result_to_batch(
                    result_path,
                    "query-checksum",
                    self._metadata(),
                    batch_size=2,
                )

                for key, replacement in (
                    ("matrix", "PAM30"),
                    ("query_chunk_sha256", "different-query"),
                    ("source_result_filename", "different-result.txt"),
                    ("source_result_sha256", "different-result-checksum"),
                ):
                    expected = dict(record["attrs"])
                    expected[key] = replacement
                    valid, reason, _ = substitution_matrix.validate_batch_file(
                        record["path"],
                        expected,
                        sequence_count=4,
                        read_size=2,
                    )
                    self.assertFalse(valid, key)
                    self.assertIn(key, reason)

    def test_changed_result_or_corrupt_batch_is_quarantined_and_rebuilt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_dir = os.path.join(temp_dir, "batches")
            os.makedirs(batch_dir)
            result_path = os.path.join(temp_dir, "result.txt")
            with open(result_path, "w", encoding="utf-8") as result:
                result.write(self._blast_line(0, 1, "1e-10"))

            with mock.patch.object(substitution_matrix, "BATCH_DIR", batch_dir):
                first = substitution_matrix.parse_result_to_batch(
                    result_path,
                    "query-checksum",
                    self._metadata(),
                    batch_size=2,
                )
                with open(result_path, "a", encoding="utf-8") as result:
                    result.write(self._blast_line(0, 2, "1e-20"))
                second = substitution_matrix.parse_result_to_batch(
                    result_path,
                    "query-checksum",
                    self._metadata(),
                    batch_size=2,
                )

                self.assertFalse(second["reused"])
                self.assertEqual(second["edges"], 2)
                invalid_dir = os.path.join(batch_dir, "invalid_batches")
                self.assertTrue(os.path.isdir(invalid_dir))

                with h5py.File(second["path"], "r+") as batch:
                    batch["j"].resize((1,))
                rebuilt = substitution_matrix.parse_result_to_batch(
                    result_path,
                    "query-checksum",
                    self._metadata(),
                    batch_size=2,
                )
                self.assertFalse(rebuilt["reused"])
                self.assertEqual(rebuilt["edges"], 2)

            self.assertNotEqual(first["attrs"]["source_result_sha256"], second["attrs"]["source_result_sha256"])

    def test_streaming_compiler_uses_only_expected_batches_and_supports_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_dir = os.path.join(temp_dir, "batches")
            os.makedirs(batch_dir)
            metadata = self._metadata()

            records = []
            with mock.patch.object(substitution_matrix, "BATCH_DIR", batch_dir):
                for query_name, lines in (
                    (
                        "query-a",
                        [
                            self._blast_line(0, 1, "1e-10"),
                            self._blast_line(0, 2, "1e-20"),
                        ],
                    ),
                    (
                        "query-b",
                        [
                            self._blast_line(1, 3, "1e-30"),
                            self._blast_line(1, 0, "1e-40"),
                        ],
                    ),
                ):
                    result_path = os.path.join(temp_dir, f"{query_name}.txt")
                    with open(result_path, "w", encoding="utf-8") as result:
                        result.writelines(lines)
                    records.append(
                        substitution_matrix.parse_result_to_batch(
                            result_path,
                            query_name,
                            metadata,
                            batch_size=1,
                        )
                    )

                # This valid extra batch must not be included unless explicitly expected.
                extra_result = os.path.join(temp_dir, "extra.txt")
                with open(extra_result, "w", encoding="utf-8") as result:
                    result.write(self._blast_line(2, 3, "1e-40"))
                substitution_matrix.parse_result_to_batch(
                    extra_result,
                    "extra-query",
                    metadata,
                    batch_size=1,
                )

                output_path = os.path.join(temp_dir, "network.h5")
                edge_count = substitution_matrix.compile_final_output(
                    ["a", "b", "c", "d"],
                    records,
                    metadata,
                    output_path,
                    batch_size=1,
                )

                empty_result = os.path.join(temp_dir, "empty.txt")
                pathlib.Path(empty_result).write_text("", encoding="utf-8")
                empty_record = substitution_matrix.parse_result_to_batch(
                    empty_result,
                    "empty-query",
                    metadata,
                    batch_size=1,
                )
                empty_output = os.path.join(temp_dir, "empty_network.h5")
                empty_count = substitution_matrix.compile_final_output(
                    ["a", "b", "c", "d"],
                    [empty_record],
                    metadata,
                    empty_output,
                    batch_size=1,
                )

            self.assertEqual(edge_count, 3)
            self.assertFalse(os.path.exists(output_path + ".partial"))
            with h5py.File(output_path, "r") as network:
                np.testing.assert_array_equal(network["i"][:], [0, 0, 1])
                np.testing.assert_array_equal(network["j"][:], [1, 2, 3])
                np.testing.assert_allclose(network["score"][:], [40.0, 20.0, 30.0])
                self.assertNotIn("blastp_dir", network.attrs)
                self.assertNotIn("blastp_executable", network.attrs)

            self.assertEqual(empty_count, 0)
            with h5py.File(empty_output, "r") as network:
                self.assertEqual(len(network["i"]), 0)
                self.assertEqual(len(network["j"]), 0)
                self.assertEqual(len(network["score"]), 0)

    def test_slice_generator_enforces_memory_bound(self):
        slices = list(substitution_matrix._bounded_slices(11, 3))
        self.assertEqual(slices, [(0, 3), (3, 6), (6, 9), (9, 11)])
        self.assertTrue(all(end - start <= 3 for start, end in slices))
        with self.assertRaises(ValueError):
            list(substitution_matrix._bounded_slices(1, 0))

    def test_blast_worker_publishes_even_an_empty_result_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "result.txt")

            def fake_run(command, **kwargs):
                partial_path = command[command.index("-out") + 1]
                pathlib.Path(partial_path).write_text("", encoding="utf-8")
                return mock.Mock(returncode=0)

            query_path = os.path.join(temp_dir, "query.fasta")
            database_path = os.path.join(temp_dir, "database")
            args = (
                query_path,
                database_path,
                "blastp",
                "BLOSUM62",
                1e300,
                output_path,
                temp_dir,
                temp_dir,
            )
            with mock.patch.object(
                substitution_matrix.subprocess,
                "run",
                side_effect=fake_run,
            ):
                status = substitution_matrix.run_alignment_worker(args)

            self.assertEqual(status, "Done")
            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(os.path.getsize(output_path), 0)
            self.assertFalse(os.path.exists(output_path + ".partial"))

            with mock.patch.object(
                substitution_matrix.subprocess,
                "run",
            ) as run:
                self.assertEqual(
                    substitution_matrix.run_alignment_worker(args),
                    "Skipped",
                )
            run.assert_not_called()

    def test_thread_change_quarantines_workspace_but_batch_size_is_irrelevant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            network_dir = os.path.join(temp_dir, "networks")
            workspace = os.path.join(network_dir, "run_temp")
            chunks = os.path.join(workspace, "chunks")
            results = os.path.join(workspace, "results")
            batches = os.path.join(workspace, "batches")
            config = os.path.join(workspace, "job_config.json")
            patches = {
                "NETWORK_DIR": network_dir,
                "SAFE_TEMP_DIR": workspace,
                "CHUNKS_DIR": chunks,
                "RESULTS_DIR": results,
                "BATCH_DIR": batches,
                "CONFIG_FILE": config,
            }
            metadata = self._metadata()
            with mock.patch.multiple(substitution_matrix, **patches):
                substitution_matrix.check_and_initialize_workspace(metadata)
                marker = os.path.join(workspace, "marker.txt")
                pathlib.Path(marker).write_text("keep", encoding="utf-8")

                # BATCH_SIZE is not part of metadata and therefore does not invalidate.
                with mock.patch.object(substitution_matrix, "BATCH_SIZE", 1):
                    self.assertTrue(
                        substitution_matrix.check_and_initialize_workspace(metadata)
                    )
                self.assertTrue(os.path.exists(marker))

                changed = dict(metadata, num_threads=3)
                self.assertFalse(
                    substitution_matrix.check_and_initialize_workspace(changed)
                )
                self.assertFalse(os.path.exists(marker))
                backups = list(pathlib.Path(network_dir).glob("run_temp_BackUp*"))
                self.assertEqual(len(backups), 1)
                self.assertTrue((backups[0] / "marker.txt").exists())

    def test_failed_compilation_retains_workspace_and_explicit_cleanup_removes_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            network_dir = os.path.join(temp_dir, "networks")
            workspace = os.path.join(network_dir, "run_temp")
            os.makedirs(workspace)
            marker = os.path.join(workspace, "resume-marker.txt")
            pathlib.Path(marker).write_text("resume", encoding="utf-8")
            output_path = os.path.join(network_dir, "network.h5")
            missing_record = {
                "path": os.path.join(workspace, "missing.h5"),
                "attrs": {},
            }

            with mock.patch.object(
                substitution_matrix,
                "NETWORK_DIR",
                network_dir,
            ), mock.patch.object(
                substitution_matrix,
                "SAFE_TEMP_DIR",
                workspace,
            ):
                with self.assertRaises(RuntimeError):
                    substitution_matrix.compile_final_output(
                        ["a", "b", "c", "d"],
                        [missing_record],
                        self._metadata(),
                        output_path,
                        batch_size=1,
                    )
                self.assertTrue(os.path.exists(marker))

                substitution_matrix.cleanup_workspace()
                self.assertFalse(os.path.exists(workspace))

    def test_mocked_end_to_end_workflow_publishes_then_cleans_workspace(self):
        class ImmediatePool:
            def __init__(self, processes):
                self.processes = processes

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            @staticmethod
            def imap(function, tasks):
                for task in tasks:
                    yield function(task)

        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_dir = os.path.join(temp_dir, "fastas")
            network_dir = os.path.join(temp_dir, "networks")
            os.makedirs(fasta_dir)
            input_path = os.path.join(fasta_dir, "input.fasta")
            pathlib.Path(input_path).write_text(
                ">a\nACDE\n>b\nACDF\n",
                encoding="utf-8",
            )
            workspace = os.path.join(network_dir, "input_[BLAST]_EValue_temp")
            output_path = os.path.join(network_dir, "input_[BLAST]_EValue.h5")

            def fake_subprocess_run(command, **kwargs):
                if os.path.basename(command[0]).startswith("makeblastdb"):
                    return mock.Mock(returncode=0, stdout="", stderr="")
                partial_result = command[command.index("-out") + 1]
                pathlib.Path(partial_result).write_text(
                    self._blast_line(0, 1, "1e-25"),
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0, stdout="", stderr="")

            patches = {
                "INPUT_FASTA": "input.fasta",
                "FULL_INPUT_FASTA": input_path,
                "SEQUENCE_SET": "input",
                "NETWORK_DIR": network_dir,
                "OUTPUT_HDF5": output_path,
                "SAFE_TEMP_DIR": workspace,
                "CHUNKS_DIR": os.path.join(workspace, "chunks"),
                "RESULTS_DIR": os.path.join(workspace, "results"),
                "BATCH_DIR": os.path.join(workspace, "batches"),
                "CONFIG_FILE": os.path.join(workspace, "job_config.json"),
                "NUM_THREADS": 1,
                "BATCH_SIZE": 1,
                "MATRIX": "BLOSUM62",
                "MAKEBLASTDB_CMD": "makeblastdb",
                "BLASTP_CMD": "blastp",
            }
            with mock.patch.multiple(
                substitution_matrix,
                **patches,
            ), mock.patch.object(
                substitution_matrix,
                "get_blastp_version",
                return_value="blastp: test",
            ), mock.patch.object(
                substitution_matrix.subprocess,
                "run",
                side_effect=fake_subprocess_run,
            ), mock.patch.object(
                substitution_matrix.multiprocessing,
                "Pool",
                ImmediatePool,
            ), mock.patch.object(
                substitution_matrix,
                "configure_runtime_paths",
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                substitution_matrix.run_workflow()

            self.assertTrue(os.path.exists(output_path))
            self.assertFalse(os.path.exists(workspace))
            with h5py.File(output_path, "r") as network:
                np.testing.assert_array_equal(network["i"][:], [0])
                np.testing.assert_array_equal(network["j"][:], [1])
                np.testing.assert_allclose(network["score"][:], [25.0])
                self.assertEqual(network.attrs["model_name"], "BLAST")
                self.assertEqual(network.attrs["matrix"], "BLOSUM62")


if __name__ == "__main__":
    unittest.main()
