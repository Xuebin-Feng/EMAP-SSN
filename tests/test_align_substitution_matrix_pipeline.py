"""Tests of Align_Substitution_Matrix (BLAST all-vs-all): FASTA sanitization,
workspace derivation, guarding and resume, BLAST workers, result-to-batch
parsing and validation, streaming compilation and the mocked end-to-end run.
"""
import glob
import io
import os
import pathlib
import sys
import tempfile
import unittest
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

# The tests package points tool imports at a missing settings file, so the
# project-root tools_settings.json (a developer's own selections) is not read.
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Substitution_Matrix as substitution_matrix


def _long_paths_supported():
    """Whether this process can open paths of MAX_PATH (260) characters or more."""
    if os.name != "nt":
        return True
    import ctypes
    try:
        enabled = ctypes.WinDLL("ntdll").RtlAreLongPathsEnabled
    except (AttributeError, OSError):
        return False
    enabled.restype = ctypes.c_ubyte
    return bool(enabled())


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
                ["a_much_longer_header", "other(one)"],
            )
            self.assertEqual(sequences, ["ACXD", "MNPXXQ"])
            with open(output_path, "r", encoding="utf-8") as safe_fasta:
                self.assertEqual(
                    safe_fasta.read(),
                    ">0\nACXD\n>1\nMNPXXQ\n",
                )

    def test_temporary_workspace_is_derived_inside_network_directory(self):
        # Derive from an explicit selection. The module-level values exist only
        # when tools_settings.json selects a FASTA, which a fresh clone lacks.
        derived = ("SEQUENCE_SET", "FULL_INPUT_FASTA", "OUTPUT_HDF5", "SAFE_TEMP_DIR",
                   "CHUNKS_DIR", "RESULTS_DIR", "BATCH_DIR", "CONFIG_FILE")
        with tempfile.TemporaryDirectory() as network_dir, mock.patch.multiple(
            substitution_matrix,
            INPUT_FASTA="proteins.fasta",
            NETWORK_DIR=network_dir,
            **{name: getattr(substitution_matrix, name) for name in derived},
        ):
            substitution_matrix.configure_runtime_paths()
            workspace = substitution_matrix.SAFE_TEMP_DIR
            self.assertEqual(
                os.path.normcase(os.path.dirname(workspace)),
                os.path.normcase(os.path.normpath(network_dir)),
            )
            self.assertEqual(os.path.basename(workspace), "proteins_[BLAST]_EValue_temp")
            for name in ("CHUNKS_DIR", "RESULTS_DIR", "BATCH_DIR", "CONFIG_FILE"):
                self.assertEqual(os.path.dirname(getattr(substitution_matrix, name)), workspace)

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

    def test_result_parser_scores_evalues_and_counts_rejected_lines(self):
        lines = [
            self._blast_line(0, 1, "0"),  # E-value 0 scores 300 (the 1e-300 floor)
            self._blast_line(2, 2, "1e-5"),  # self-hit: valid, but no edge
            self._blast_line(3, 1, "1e-4"),  # stored as (1, 3)
            "# BLAST comment\n",
            "\n",
            "0 1 100.0 10\n",  # too few columns
            self._blast_line("x", 1, "1e-5"),  # non-integer index
            self._blast_line(0, 4, "1e-5"),  # index beyond the four sequences
            self._blast_line(-1, 2, "1e-5"),  # negative index
            self._blast_line(1, 2, "nan"),  # non-finite E-values
            self._blast_line(1, 2, "inf"),
            self._blast_line(1, 2, "-1e-5"),  # negative E-value
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_dir = os.path.join(temp_dir, "batches")
            os.makedirs(batch_dir)
            result_path = os.path.join(temp_dir, "result.txt")
            with open(result_path, "w", encoding="utf-8") as result:
                result.writelines(lines)

            with mock.patch.object(substitution_matrix, "BATCH_DIR", batch_dir):
                record = substitution_matrix.parse_result_to_batch(
                    result_path,
                    "query-checksum",
                    self._metadata(),
                    batch_size=2,
                )

            self.assertEqual(
                (record["edges"], record["valid_lines"], record["mismatches"]),
                (2, 3, 7),
            )
            with h5py.File(record["path"], "r") as batch:
                np.testing.assert_array_equal(batch["i"][:], [0, 1])
                np.testing.assert_array_equal(batch["j"][:], [1, 3])
                np.testing.assert_allclose(batch["score"][:], [300.0, 4.0], rtol=1e-6)

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

    def test_blast_workspace_alias_is_itself_short_and_space_free(self):
        # The length limit applies on Windows only; patching it in checks the
        # rules on every platform. Nothing here touches the disk.
        limit = 260
        self.assertEqual(
            substitution_matrix.BLAST_MAX_PATH,
            limit if os.name == "nt" else None,
        )
        reserve = substitution_matrix.BLAST_PATH_RESERVE
        root = os.path.abspath(os.sep)
        link_base = os.path.join(root, "links")

        def path_of_length(length, fill="w"):
            return os.path.join(root, fill * (length - len(root)))

        at_limit = path_of_length(limit - reserve)
        cases = (
            # label, workspace, what _windows_short_path returns, expected alias
            ("below the limit", path_of_length(limit - reserve - 1), None, "workspace"),
            # A volume without 8.3 names returns the long path unchanged.
            ("at the limit without short names", at_limit, at_limit, "link"),
            ("at the limit with a short path", at_limit, os.path.join(root, "SHORT~1"), "short path"),
            ("containing a space", os.path.join(root, "My Networks", "set_temp"), None, "link"),
        )
        for label, workspace, short_path, expected in cases:
            create_link = mock.Mock()
            remove_link = mock.Mock()
            with self.subTest(label), mock.patch.multiple(
                substitution_matrix,
                BLAST_MAX_PATH=limit,
                _windows_short_path=mock.Mock(return_value=short_path),
                _space_free_link_bases=mock.Mock(return_value=[link_base]),
                _create_directory_link=create_link,
                _remove_directory_link=remove_link,
            ), redirect_stdout(io.StringIO()):
                alias, release = substitution_matrix.resolve_blast_workspace(workspace)
                release()

                if expected == "workspace":
                    self.assertEqual(alias, workspace)
                elif expected == "short path":
                    self.assertEqual(alias, short_path)
                else:
                    self.assertEqual(os.path.dirname(alias), link_base)
                    create_link.assert_called_once_with(alias, workspace)
                    remove_link.assert_called_once_with(alias)
                if expected != "link":
                    create_link.assert_not_called()
                self.assertNotIn(" ", alias)
                self.assertLess(len(alias) + reserve, limit)

        # A link under a base that is itself too long would fail the same way,
        # so it is never created and the error reports the length.
        create_link = mock.Mock()
        with mock.patch.multiple(
            substitution_matrix,
            BLAST_MAX_PATH=limit,
            _windows_short_path=mock.Mock(return_value=None),
            _space_free_link_bases=mock.Mock(return_value=[at_limit]),
            _create_directory_link=create_link,
        ), self.assertRaises(RuntimeError) as raised:
            substitution_matrix.resolve_blast_workspace(at_limit)
        create_link.assert_not_called()
        message = str(raised.exception)
        self.assertIn(f"is too long for BLAST ({limit - reserve} characters", message)
        self.assertIn(f"{at_limit}: the link path would be too long", message)

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

    def test_workspace_operations_refuse_paths_outside_network_directory(self):
        # cleanup_workspace removes SAFE_TEMP_DIR with rmtree, so the guard must
        # refuse the network folder itself and every folder outside it.
        with tempfile.TemporaryDirectory() as temp_dir:
            network_dir = os.path.join(temp_dir, "networks")
            cases = (
                ("network folder", network_dir),
                ("parent folder", temp_dir),
                ("sibling sharing the name prefix", network_dir + "_temp"),
                ("path leaving through ..", os.path.join(network_dir, "..", "elsewhere")),
            )
            operations = (
                ("cleanup", substitution_matrix.cleanup_workspace),
                ("quarantine", lambda: substitution_matrix.quarantine_workspace("test")),
                ("initialize", lambda: substitution_matrix.check_and_initialize_workspace(
                    self._metadata()
                )),
            )
            markers = []
            for label, workspace in cases:
                os.makedirs(workspace, exist_ok=True)
                marker = os.path.join(workspace, f"marker_{len(markers)}.txt")
                pathlib.Path(marker).write_text(label, encoding="utf-8")
                markers.append(marker)
                for name, operation in operations:
                    with self.subTest(workspace=label, operation=name), mock.patch.multiple(
                        substitution_matrix,
                        NETWORK_DIR=network_dir,
                        SAFE_TEMP_DIR=workspace,
                    ), redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(
                            RuntimeError, "Refusing workspace operation outside NETWORK_DIR"
                        ):
                            operation()

            for marker in markers:
                self.assertTrue(os.path.exists(marker), marker)
            self.assertEqual(
                glob.glob(os.path.join(temp_dir, "**", "*_BackUp*"), recursive=True), []
            )

            # Control: a workspace inside NETWORK_DIR is removed, the folder kept.
            inside = os.path.join(network_dir, "set_[BLAST]_EValue_temp")
            os.makedirs(inside)
            with mock.patch.multiple(
                substitution_matrix, NETWORK_DIR=network_dir, SAFE_TEMP_DIR=inside
            ):
                substitution_matrix.cleanup_workspace()
            self.assertFalse(os.path.exists(inside))
            self.assertTrue(os.path.exists(markers[0]))

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

    def _run_mocked_workflow(self, input_path, network_dir, fake_subprocess_run, **patches):
        """Run main([]) for input_path with BLAST, the worker pool and settings mocked.

        Returns the exit code and the mocked multiprocessing.set_start_method.
        """
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

        workspace = os.path.join(network_dir, "input_[BLAST]_EValue_temp")
        settings = {
            "INPUT_FASTA": "input.fasta",
            "FULL_INPUT_FASTA": input_path,
            "SEQUENCE_SET": "input",
            "NETWORK_DIR": network_dir,
            "OUTPUT_HDF5": os.path.join(network_dir, "input_[BLAST]_EValue.h5"),
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
        settings.update(patches)
        with mock.patch.multiple(
            substitution_matrix,
            **settings,
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
        ), mock.patch.object(
            # main([]) reads the project-root tools_settings.json (a
            # developer's own selections); the import hook doesn't cover it.
            substitution_matrix,
            "load_tool_settings",
        ), mock.patch.object(
            # On Linux this would switch the whole test process to spawn.
            substitution_matrix.multiprocessing,
            "set_start_method",
        ) as set_start_method, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exit_code = substitution_matrix.main([])
        return exit_code, set_start_method

    def test_mocked_end_to_end_workflow_publishes_then_cleans_workspace(self):
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

            exit_code, set_start_method = self._run_mocked_workflow(
                input_path,
                network_dir,
                fake_subprocess_run,
            )

            # main()'s return value is the process exit code, and the MCP job
            # runner reports a job as succeeded only for exit code 0.
            self.assertEqual(exit_code, 0)
            set_start_method.assert_called_once_with("spawn")

            self.assertTrue(os.path.exists(output_path))
            self.assertFalse(os.path.exists(workspace))
            with h5py.File(output_path, "r") as network:
                np.testing.assert_array_equal(network["i"][:], [0])
                np.testing.assert_array_equal(network["j"][:], [1])
                np.testing.assert_allclose(network["score"][:], [25.0])
                self.assertEqual(network.attrs["model_name"], "BLAST")
                self.assertEqual(network.attrs["matrix"], "BLOSUM62")

    @unittest.skipUnless(
        _long_paths_supported(),
        "Windows long paths are disabled, so Python cannot write this workspace either",
    )
    def test_deep_space_free_workspace_keeps_blast_paths_under_max_path(self):
        # An MCP job's space-free NETWORK_DIR made the workspace 172 characters
        # long, and blastp, which is not long-path aware, rejected its
        # 264-character -out path as "not accessible".
        limit = 260
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_dir = os.path.join(temp_dir, "fastas")
            os.makedirs(fasta_dir)
            input_path = os.path.join(fasta_dir, "input.fasta")
            pathlib.Path(input_path).write_text(
                ">a\nACDE\n>b\nACDF\n",
                encoding="utf-8",
            )
            workspace_name = "input_[BLAST]_EValue_temp"
            padding = "p" * max(1, 172 - len(temp_dir) - len(workspace_name) - 2)
            network_dir = os.path.join(temp_dir, padding)
            workspace = os.path.abspath(os.path.join(network_dir, workspace_name))
            longest_name = os.path.join("results", "result_" + "0" * 64 + ".txt.partial")
            self.assertGreaterEqual(len(os.path.join(workspace, longest_name)), limit)

            links = {}
            commands = []

            def create_link(link_path, target_path):
                links[link_path] = target_path

            def through_links(path):
                # Resolve the mocked junction the way Windows would.
                for link_path, target_path in links.items():
                    if path.startswith(link_path + os.sep):
                        return target_path + path[len(link_path):]
                return path

            def fake_subprocess_run(command, **kwargs):
                commands.append((list(command), kwargs.get("cwd")))
                if os.path.basename(command[0]).startswith("makeblastdb"):
                    return mock.Mock(returncode=0, stdout="", stderr="")
                partial_result = through_links(command[command.index("-out") + 1])
                pathlib.Path(partial_result).write_text(
                    self._blast_line(0, 1, "1e-25"),
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0, stdout="", stderr="")

            remove_link = mock.Mock()
            exit_code, _ = self._run_mocked_workflow(
                input_path,
                network_dir,
                fake_subprocess_run,
                BLAST_MAX_PATH=limit,
                # A volume without 8.3 names returns the long path unchanged.
                _windows_short_path=lambda path: path,
                _create_directory_link=create_link,
                _remove_directory_link=remove_link,
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(
                os.path.exists(os.path.join(network_dir, "input_[BLAST]_EValue.h5"))
            )
            self.assertFalse(os.path.exists(workspace))

            self.assertEqual(
                [os.path.basename(command[0]) for command, _ in commands],
                ["makeblastdb", "blastp"],
            )
            blast_paths = []
            for command, cwd in commands:
                blast_paths.append(cwd)
                blast_paths.extend(
                    command[index + 1]
                    for index, argument in enumerate(command[:-1])
                    if argument in ("-in", "-out", "-query", "-db")
                )
            for path in blast_paths:
                with self.subTest(path=path):
                    self.assertLess(len(path), limit)

            # BLAST reached the workspace through one link, released afterwards.
            self.assertEqual(list(links.values()), [workspace])
            (link,) = links
            remove_link.assert_called_once_with(link)
            for path in blast_paths:
                self.assertTrue(path == link or path.startswith(link + os.sep), path)

    def test_main_exits_nonzero_when_no_fasta_is_selected(self):
        # The MCP job runner reports exit code 0 as a successful job, so a run
        # without a selected FASTA reached agents as a success that wrote nothing.
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = io.StringIO()
            with mock.patch.multiple(
                substitution_matrix,
                INPUT_FASTA=None,
                FASTA_DIR=os.path.join(temp_dir, "fastas"),
                NETWORK_DIR=os.path.join(temp_dir, "networks"),
            ), mock.patch.object(
                # main([]) reads the project-root tools_settings.json (a
                # developer's own selections); the import hook doesn't cover it.
                substitution_matrix,
                "load_tool_settings",
            ), mock.patch.object(
                # On Linux this would switch the whole test process to spawn.
                substitution_matrix.multiprocessing,
                "set_start_method",
            ), redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                exit_code = substitution_matrix.main([])

            self.assertEqual(exit_code, 1)
            self.assertIn(
                "❌ Error: No input FASTA file has been selected.",
                stdout.getvalue(),
            )
            self.assertEqual(os.listdir(temp_dir), [])


if __name__ == "__main__":
    unittest.main()
