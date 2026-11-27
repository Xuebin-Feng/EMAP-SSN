"""Tests of Align_Similarity_Matrix resume scans, partial-batch checkpoints, and network compilation."""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import h5py
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
UTILITIES_DIR = os.path.join(SRC_DIR, "utilities")
TOOLS_DIR = os.path.join(SRC_DIR, "tools")
for _folder in (SRC_DIR, UTILITIES_DIR, TOOLS_DIR):
    if _folder not in sys.path:
        sys.path.insert(0, _folder)

# The tests package points tool imports at a missing settings file.
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as similarity_matrix

from tests.align_pipeline_fixtures import (  # noqa: E402
    COMPILE_BATCHES,
    OLD_ATTRIBUTES,
    OLD_DATASETS,
    compile_batches,
    run_align_main,
    write_compile_batches,
)
from tests.alignment_fixtures import BatchResumeScanFixture  # noqa: E402


class AlignmentResumeAndCompileTests(unittest.TestCase):
    def test_confirmation_sample_matches_production_pair_order(self):
        headers = ["a", "b", "c", "d"]
        computed = np.zeros((4, 4), dtype=bool)
        computed[0, 2] = True
        sample = similarity_matrix._first_pending_pairs(
            headers,
            computed,
            num_tasks=5,
            limit=3,
        )
        self.assertEqual(
            [(row, column) for row, column, _left, _right in sample],
            [(0, 1), (0, 3), (1, 2)],
        )
        pending_keys = np.array([0 * 4 + 1, 0 * 4 + 3, 1 * 4 + 2], dtype=np.int64)
        sparse_sample = list(
            similarity_matrix._iter_pending_key_tasks(headers, pending_keys, 4)
        )
        self.assertEqual(
            [(row, column) for row, column, _left, _right in sparse_sample],
            [(0, 1), (0, 3), (1, 2)],
        )

    def test_sparse_resume_counts_upper_triangle_pairs_once(self):
        upper = np.triu(np.ones((4, 4), dtype=bool), k=1)
        computed = upper | upper.T

        self.assertEqual(
            similarity_matrix._count_computed_required_pairs(computed),
            6,
        )

        computed[1, 3] = False
        computed[3, 1] = False
        self.assertEqual(
            similarity_matrix._count_computed_required_pairs(computed),
            5,
        )

        # Sparse key resume deduplicates against required_keys without N x N masks.
        required_keys = np.array([1, 2, 3, 6, 7, 11], dtype=np.int64)
        computed_keys = np.array([1, 2, 3, 6, 11], dtype=np.int64)
        already_done = similarity_matrix._locate_sorted_keys(computed_keys, required_keys)[1]
        self.assertEqual(int(np.count_nonzero(already_done)), 5)
        np.testing.assert_array_equal(required_keys[~already_done], np.array([7], dtype=np.int64))

    def test_legacy_precision_cache_resumes_as_fp32_and_tf32_mismatch_backs_up(self):
        def write_batch(folder):
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, "batch_00000.h5")
            with h5py.File(path, "w") as hf:
                hf.attrs["embedding_checksum"] = "checksum"
                hf.attrs["model_name"] = "model"
                hf.attrs["gap_penalties"] = np.array([-2.0, 0.0], np.float32)
                for name, values, dtype in (
                    ("i", [0], np.uint32),
                    ("j", [1], np.uint32),
                    ("l_score", [1.0], np.float32),
                    ("l_len", [1], np.uint16),
                    ("g_score", [1.0], np.float32),
                    ("g_len", [1], np.uint16),
                ):
                    hf.create_dataset(name, data=np.asarray(values, dtype=dtype))

        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = os.path.join(temp_dir, "batches")
            write_batch(results_dir)
            computed = np.zeros((2, 2), dtype=bool)
            with mock.patch.object(
                similarity_matrix, "RESULTS_DIR", results_dir
            ):
                precision = similarity_matrix.scan_existing_batches(
                    2,
                    "checksum",
                    "model",
                    "float32",
                    [-2.0, 0.0],
                    computed,
                )
            self.assertEqual(precision, "ieee_fp32")
            self.assertTrue(computed[0, 1])

            computed.fill(False)
            with mock.patch.object(
                similarity_matrix, "RESULTS_DIR", results_dir
            ), redirect_stdout(io.StringIO()):
                precision = similarity_matrix.scan_existing_batches(
                    2,
                    "checksum",
                    "model",
                    "float32",
                    [-2.0, 0.0],
                    computed,
                    requested_matmul_precision="tf32",
                )
            self.assertIsNone(precision)
            self.assertTrue(os.path.isdir(results_dir))
            self.assertTrue(os.path.isdir(results_dir + "_BackUp"))

    def test_partial_batch_writer_rolls_back_before_atomic_publish(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "batch_00000.h5")
            writer = similarity_matrix._PartialBatchWriter(
                output,
                "checksum",
                "model",
                [-2.0, 0.0],
                "ieee_fp32",
            )
            writer(
                [
                    (0, 1, 1.0, 1, 2.0, 1),
                    (0, 2, 1.5, 2, 2.5, 2),
                ]
            )
            checkpoint = writer.checkpoint()
            writer([(1, 2, 3.0, 3, 4.0, 3)])
            writer.rollback(checkpoint)
            writer.publish()
            writer.close()

            self.assertFalse(os.path.exists(output + ".partial"))
            with h5py.File(output, "r") as hf:
                np.testing.assert_array_equal(hf["i"][:], [0, 0])
                np.testing.assert_array_equal(hf["j"][:], [1, 2])

    def test_compile_writes_the_required_pairs_and_the_old_metadata_only(self):
        required_keys = np.array(
            [r * 6 + c for r, c in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4)]],
            dtype=np.int64,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            write_compile_batches(os.path.join(temp_dir, "batches"))
            output, _ = compile_batches(temp_dir, required_keys=required_keys)
            with h5py.File(output, "r") as hf:
                edges = set(zip(hf["i"][:].tolist(), hf["j"][:].tolist()))
                attributes, datasets = set(hf.attrs), set(hf)
                self.assertIsNone(similarity_matrix.Prefilter_Calibration.read_record(hf))
            self.assertIn((3, 4), edges)
            self.assertNotIn((4, 5), edges)
            # What a network carried before the prefilter stored anything: no prefilter_* attribute, no calibration dataset.
            self.assertEqual(attributes, OLD_ATTRIBUTES)
            self.assertEqual(datasets, OLD_DATASETS)

    def test_compile_publishes_complete_network_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_compile_batches(os.path.join(temp_dir, "batches"))
            output, stdout = compile_batches(temp_dir)

            self.assertEqual(
                os.listdir(os.path.dirname(output)),
                [os.path.basename(output)],
            )
            self.assertIn("✅ Compilation complete!", stdout)
            pairs = [
                pair
                for batch in COMPILE_BATCHES.values()
                for pair in batch
            ]
            with h5py.File(output, "r") as hf:
                np.testing.assert_array_equal(hf["i"][:], [p[0] for p in pairs])
                np.testing.assert_array_equal(hf["j"][:], [p[1] for p in pairs])
                np.testing.assert_array_equal(
                    hf["l_score"][:],
                    [p[0] * 10 + p[1] + 0.5 for p in pairs],
                )
                np.testing.assert_array_equal(
                    hf["g_len"][:],
                    [p[0] + p[1] + 2 for p in pairs],
                )

    def test_corrupt_batch_fails_compilation_without_a_network_file(self):
        # A batch missing a score column used to leave its i/j rows (or zero
        # fill) behind zero scores, and an unreadable one silently dropped its
        # edges; both then printed "Compilation complete".
        cases = (
            ("batch_00001.h5", "not_hdf5", "Error reading .*batch_00001.h5 during counting"),
            ("batch_00001.h5", "missing_l_score", "Error merging batch .*batch_00001.h5"),
            ("batch_00002.h5", "missing_l_score", "Error merging batch .*batch_00002.h5"),
        )
        for corrupt, corruption, message in cases:
            with self.subTest(batch=corrupt, corruption=corruption):
                with tempfile.TemporaryDirectory() as temp_dir:
                    write_compile_batches(
                        os.path.join(temp_dir, "batches"),
                        corrupt,
                        corruption,
                    )
                    output = os.path.join(
                        temp_dir, "network", "set_[model]_network.h5"
                    )
                    with self.assertRaisesRegex(
                        similarity_matrix.NetworkCompilationError,
                        message,
                    ):
                        compile_batches(temp_dir)

                    self.assertFalse(os.path.exists(output))
                    self.assertFalse(os.path.exists(output + ".partial"))

    def test_main_exits_nonzero_when_a_batch_cannot_be_merged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_compile_batches(
                os.path.join(temp_dir, "batches"),
                "batch_00001.h5",
                "missing_l_score",
            )
            stdout = io.StringIO()
            with mock.patch.object(
                similarity_matrix,
                "load_tool_settings",
            ), mock.patch.object(
                similarity_matrix,
                "run_job_distributor",
                side_effect=lambda: compile_batches(temp_dir),
            ), redirect_stdout(stdout):
                exit_code = similarity_matrix.main([])

            self.assertEqual(exit_code, 1)
            self.assertIn("❌ Cannot compile the final network", stdout.getvalue())
            self.assertIn("batch_00001.h5", stdout.getvalue())
            self.assertFalse(
                os.path.exists(os.path.join(temp_dir, "network", "set_[model]_network.h5"))
            )

    def test_main_exit_code_reflects_the_existing_network_check(self):
        cases = (
            ("done", "ieee_fp32", "automatic_32bit", 0, "✅ Job already done."),
            ("bf16 vs auto", "bf16", "automatic_32bit", 1,
             "❌ Completed output precision conflict: found 'bf16'"),
            ("tf32 vs float32", "tf32", "float32", 1,
             "❌ Completed output precision conflict: found 'tf32'"),
            ("unreadable", None, "automatic_32bit", 1,
             "❌ Cannot inspect completed output precision"),
        )
        for label, stored, requested, expected_code, message in cases:
            with self.subTest(label), tempfile.TemporaryDirectory() as temp_dir:
                network = os.path.join(temp_dir, "network", "set_[model]_network.h5")
                os.makedirs(os.path.dirname(network))
                if stored is None:
                    with open(network, "wb") as handle:
                        handle.write(b"truncated network")
                else:
                    with h5py.File(network, "w") as hf:
                        hf.attrs["matmul_precision"] = stored
                with open(network, "rb") as handle:
                    original = handle.read()

                exit_code, stdout, compile_output = run_align_main(
                    temp_dir, ACCELERATOR_PRECISION=requested
                )

                self.assertEqual(exit_code, expected_code)
                self.assertIn(message, stdout)
                compile_output.assert_not_called()
                with open(network, "rb") as handle:
                    self.assertEqual(handle.read(), original)


class AlignmentResumeScanTests(BatchResumeScanFixture, unittest.TestCase):
    """scan_existing_batches resumes only batches written for this run.

    Any other batch sends the whole folder to <RESULTS_DIR>_BackUp with a
    report, and the run starts again from an empty folder.
    """

    MATCHING_ATTRS = {
        "embedding_checksum": "checksum",
        "model_name": "model",
        "gap_penalties": np.array([-2.0, 0.0], np.float32),
        "matmul_precision": "ieee_fp32",
    }

    def scan(self, results_dir, saving_mode="float32"):
        computed = np.zeros((2, 2), dtype=bool)
        stdout = io.StringIO()
        with mock.patch.object(
            similarity_matrix, "RESULTS_DIR", results_dir
        ), redirect_stdout(stdout):
            precision = similarity_matrix.scan_existing_batches(
                2, "checksum", "model", saving_mode, [-2.0, 0.0], computed
            )
        return precision, computed, stdout.getvalue()

    def test_matching_batch_resumes_its_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = os.path.join(temp_dir, "batches")
            self.write_batch(results_dir)
            # Batches record no saving mode; the embeddings checksum changes
            # with it instead.
            for saving_mode in ("float32", "float16"):
                with self.subTest(saving_mode=saving_mode):
                    precision, computed, _output = self.scan(results_dir, saving_mode)
                    self.assertEqual(precision, "ieee_fp32")
                    np.testing.assert_array_equal(
                        computed, [[False, True], [True, False]]
                    )
            self.assertEqual(os.listdir(temp_dir), ["batches"])
            self.assertEqual(os.listdir(results_dir), ["batch_00000.h5"])

    def test_each_identity_mismatch_backs_up_the_folder(self):
        cached_gaps = list(np.array([-3.0, 0.0], np.float32))
        cases = {
            "checksum": (
                {"embedding_checksum": "other"},
                "Checksum mismatch in 'batch_00000.h5' ('other' vs current 'checksum')",
            ),
            "model_name": (
                {"model_name": "other-model"},
                "Model name mismatch in 'batch_00000.h5' ('other-model' vs current 'model')",
            ),
            "gap_penalties": (
                {"gap_penalties": np.array([-3.0, 0.0], np.float32)},
                f"Gap penalties mismatch in 'batch_00000.h5' ({cached_gaps} vs current [-2.0, 0.0])",
            ),
            "missing l_len": (
                {"omit": ("l_len",)},
                "Missing required datasets in batch file 'batch_00000.h5'",
            ),
            "i/j length": (
                {"j": (1, 1)},
                "Dataset length mismatch in batch file 'batch_00000.h5'",
            ),
        }
        for label, (batch, reason) in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as temp_dir:
                results_dir = os.path.join(temp_dir, "batches")
                self.write_batch(results_dir, **batch)
                precision, computed, output = self.scan(results_dir)
                reasons = self.assert_backed_up(results_dir, results_dir + "_BackUp")
                self.assertIsNone(precision)
                self.assertFalse(computed.any())
                self.assertEqual(reasons, [reason])
                self.assertIn(f"  > Mismatch reason: {reason}\n", output)

    def test_mixed_precision_batches_back_up_the_folder(self):
        # Each batch alone would resume: no precision is requested here.
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = os.path.join(temp_dir, "batches")
            self.write_batch(results_dir, "batch_00000.h5")
            self.write_batch(results_dir, "batch_00001.h5", matmul_precision="tf32")
            precision, computed, _output = self.scan(results_dir)
            reasons = self.assert_backed_up(
                results_dir,
                results_dir + "_BackUp",
                batches=("batch_00000.h5", "batch_00001.h5"),
            )
        self.assertIsNone(precision)
        self.assertFalse(computed.any())
        # The glob order decides which batch is read second.
        self.assertIn(
            reasons,
            [
                ["Mixed matmul precision in 'batch_00001.h5' ('tf32' vs 'ieee_fp32')"],
                ["Mixed matmul precision in 'batch_00000.h5' ('ieee_fp32' vs 'tf32')"],
            ],
        )

    def test_repeat_backup_takes_the_next_free_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = os.path.join(temp_dir, "batches")
            self.write_batch(results_dir, model_name="first")
            self.scan(results_dir)
            self.write_batch(results_dir, model_name="second")
            precision, computed, output = self.scan(results_dir)
            reasons = self.assert_backed_up(results_dir, results_dir + "_BackUp_1")
            self.assertEqual(
                sorted(os.listdir(temp_dir)),
                ["batches", "batches_BackUp", "batches_BackUp_1"],
            )
            first_batch = os.path.join(results_dir + "_BackUp", "batch_00000.h5")
            with h5py.File(first_batch, "r") as hf:
                self.assertEqual(hf.attrs["model_name"], "first")
        self.assertIsNone(precision)
        self.assertFalse(computed.any())
        self.assertEqual(
            reasons,
            ["Model name mismatch in 'batch_00000.h5' ('second' vs current 'model')"],
        )
        self.assertIn(
            f"  > Renaming existing batch folder to: '{results_dir}_BackUp_1'\n",
            output,
        )


if __name__ == "__main__":
    unittest.main()
