import io
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import h5py
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILITIES_DIR = os.path.join(PROJECT_ROOT, "src", "utilities")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "src", "tools")
if UTILITIES_DIR not in sys.path:
    sys.path.insert(0, UTILITIES_DIR)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as similarity_matrix
    import Embedding_HDF5


class ImmediateExecutor:
    instances = []

    def __init__(self, max_workers, **kwargs):
        self.max_workers = max_workers
        self.options = kwargs
        self.submissions = []
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def submit(self, function, *args):
        self.submissions.append((function, args))
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as error:
            future.set_exception(error)
        return future


class AlignmentPipelineTests(unittest.TestCase):
    def setUp(self):
        similarity_matrix.std_mean_support_cache.clear()

    def test_runtime_path_configuration_preserves_none_as_unselected(self):
        with mock.patch.object(similarity_matrix, "INPUT_HDF5", None):
            with self.assertRaisesRegex(ValueError, "No embeddings file"):
                similarity_matrix.configure_runtime_paths()

    def test_runtime_path_configuration_accepts_an_absolute_input(self):
        selected = os.path.abspath(
            os.path.join("temporary", "sequences_[test-model]_embeddings.h5")
        )
        with mock.patch.object(similarity_matrix, "INPUT_HDF5", selected), \
                mock.patch.object(similarity_matrix, "FULL_INPUT_HDF5"), \
                mock.patch.object(similarity_matrix, "SEQUENCE_SET"), \
                mock.patch.object(similarity_matrix, "MODEL_NAME"), \
                mock.patch.object(similarity_matrix, "RESULTS_DIR"), \
                mock.patch.object(similarity_matrix, "FINAL_OUTPUT_NET"):
            similarity_matrix.configure_runtime_paths()
            self.assertEqual(similarity_matrix.FULL_INPUT_HDF5, selected)
            self.assertEqual(similarity_matrix.SEQUENCE_SET, "sequences")
            self.assertEqual(similarity_matrix.MODEL_NAME, "test-model")

    def test_embedding_metadata_loader_rejects_unfinalized_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, "incomplete.h5")
            with h5py.File(h5_path, "w") as hf:
                Embedding_HDF5.create_metadata_first_file(
                    hf,
                    ["first"],
                    ["ACD"],
                    "test-model",
                    "float32",
                )

            with self.assertRaisesRegex(
                similarity_matrix.EmbeddingFileError,
                "Embedding generation is incomplete",
            ):
                similarity_matrix.load_embedding_metadata(h5_path)

    def test_embedding_metadata_loader_preserves_header_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, "complete.h5")
            with h5py.File(h5_path, "w") as hf:
                group = Embedding_HDF5.create_metadata_first_file(
                    hf,
                    ["second_header", "first"],
                    ["AC", "ACD"],
                    "test-model",
                    "float32",
                )
                group.create_dataset(
                    "first",
                    data=np.ones((3, 4), dtype=np.float32),
                )
                group.create_dataset(
                    "second_header",
                    data=np.ones((2, 4), dtype=np.float32),
                )
                Embedding_HDF5.mark_generation_complete(hf)

            headers, safe_headers, lengths, manifest = (
                similarity_matrix.load_embedding_metadata(h5_path)
            )

        self.assertEqual(headers, ["second_header", "first"])
        self.assertEqual(safe_headers, ["second_header", "first"])
        self.assertEqual(lengths, [2, 3])
        self.assertEqual(manifest.headers, headers)

    def test_cpu_alignment_worker_preserves_scores_and_lengths(self):
        matrix = np.array(
            [
                [3.0, -1.0],
                [-1.0, 3.0],
            ],
            dtype=np.float32,
        )

        result = similarity_matrix.calculate_alignment_data(
            (4, 7, matrix)
        )

        self.assertEqual(result[:2], (4, 7))
        self.assertEqual(float(result[2]), 2.0)
        self.assertEqual(int(result[3]), 2)
        self.assertEqual(float(result[4]), 6.0)
        self.assertEqual(int(result[5]), 2)

    def test_accelerated_pipeline_separates_gpu_and_cpu_stages(self):
        ImmediateExecutor.instances.clear()
        tasks = [
            (0, 1, "a", "b"),
            (0, 2, "a", "c"),
        ]
        score_calls = []
        row_embedding_ids = []

        def fake_score_matrix(emb_i, emb_j, device):
            score_calls.append((emb_i.copy(), emb_j.copy(), device))
            row_embedding_ids.append(id(emb_i))
            matrix = np.full(
                (emb_i.shape[0], emb_j.shape[0]),
                3.0,
                dtype=np.float32,
            )
            return matrix

        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, "embeddings.h5")
            with h5py.File(h5_path, "w") as hf:
                group = hf.create_group("embeddings")
                group.create_dataset(
                    "a",
                    data=np.ones((2, 4), dtype=np.float32),
                )
                group.create_dataset(
                    "b",
                    data=np.ones((3, 4), dtype=np.float32),
                )
                group.create_dataset(
                    "c",
                    data=np.ones((2, 4), dtype=np.float32),
                )

            with mock.patch.object(
                similarity_matrix,
                "ThreadPoolExecutor",
                ImmediateExecutor,
            ), mock.patch.object(
                similarity_matrix,
                "compute_score_matrix_torch",
                side_effect=fake_score_matrix,
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                results = similarity_matrix.process_accelerated_tasks(
                    tasks,
                    workers=2,
                    input_h5=h5_path,
                    device=torch.device("mps"),
                    batch_id=3,
                )

        self.assertEqual(len(results), 2)
        self.assertEqual(len(score_calls), 2)
        self.assertEqual(len(set(row_embedding_ids)), 1)
        self.assertEqual(len(ImmediateExecutor.instances), 2)
        gpu_executor, cpu_executor = ImmediateExecutor.instances
        self.assertEqual(gpu_executor.max_workers, 1)
        self.assertEqual(cpu_executor.max_workers, 2)
        self.assertEqual(len(gpu_executor.submissions), 2)
        self.assertEqual(len(cpu_executor.submissions), 2)
        self.assertTrue(
            all(
                function is similarity_matrix.calculate_alignment_data
                for function, _ in cpu_executor.submissions
            )
        )

    def test_accelerator_lane_reuses_normalized_row_until_row_changes(self):
        row_a = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
        row_b = np.arange(8, dtype=np.float32).reshape(2, 4) + 2.0
        target_a = np.ones((2, 4), dtype=np.float32)
        target_b = np.ones((4, 4), dtype=np.float32)
        target_c = np.ones((3, 4), dtype=np.float32)
        original_normalize = similarity_matrix._normalize_embedding_torch

        with mock.patch.object(
            similarity_matrix,
            "accelerator_thread_state",
            threading.local(),
        ), mock.patch.object(
            similarity_matrix,
            "_normalize_embedding_torch",
            wraps=original_normalize,
        ) as normalize:
            similarity_matrix._compute_accelerated_matrix(
                (0, 1, row_a, target_a, torch.device("cpu"))
            )
            similarity_matrix._compute_accelerated_matrix(
                (0, 2, row_a, target_b, torch.device("cpu"))
            )
            similarity_matrix._compute_accelerated_matrix(
                (1, 2, row_b, target_c, torch.device("cpu"))
            )

        normalized_inputs = [call.args[0] for call in normalize.call_args_list]
        self.assertEqual(sum(value is row_a for value in normalized_inputs), 1)
        self.assertEqual(sum(value is row_b for value in normalized_inputs), 1)
        self.assertEqual(len(normalized_inputs), 5)

    def test_accelerator_row_cache_is_isolated_per_worker_thread(self):
        row = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
        targets = [
            np.ones((2, 4), dtype=np.float32),
            np.ones((4, 4), dtype=np.float32),
        ]
        barrier = threading.Barrier(2)
        original_normalize = similarity_matrix._normalize_embedding_torch

        def run_lane(lane_number):
            barrier.wait()
            for offset, target in enumerate(targets):
                similarity_matrix._compute_accelerated_matrix(
                    (
                        0,
                        lane_number * 10 + offset,
                        row,
                        target,
                        torch.device("cpu"),
                    )
                )

        with mock.patch.object(
            similarity_matrix,
            "accelerator_thread_state",
            threading.local(),
        ), mock.patch.object(
            similarity_matrix,
            "_normalize_embedding_torch",
            wraps=original_normalize,
        ) as normalize, ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_lane, lane) for lane in range(2)]
            for future in futures:
                future.result()

        normalized_inputs = [call.args[0] for call in normalize.call_args_list]
        self.assertEqual(sum(value is row for value in normalized_inputs), 2)
        self.assertEqual(len(normalized_inputs), 6)

    def test_direct_cpu_scoring_does_not_use_accelerator_row_cache(self):
        row = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
        target = np.ones((2, 4), dtype=np.float32)
        original_normalize = similarity_matrix._normalize_embedding_torch

        with mock.patch.object(
            similarity_matrix,
            "accelerator_thread_state",
            threading.local(),
        ), mock.patch.object(
            similarity_matrix,
            "_normalize_embedding_torch",
            wraps=original_normalize,
        ) as normalize:
            similarity_matrix.compute_score_matrix_torch(
                row,
                target,
                torch.device("cpu"),
            )
            similarity_matrix.compute_score_matrix_torch(
                row,
                target,
                torch.device("cpu"),
            )

        normalized_inputs = [call.args[0] for call in normalize.call_args_list]
        self.assertEqual(sum(value is row for value in normalized_inputs), 2)
        self.assertEqual(len(normalized_inputs), 4)

    def test_fused_statistics_match_legacy_scores_and_alignment_lengths(self):
        rng = np.random.default_rng(20260801)
        device = torch.device("cpu")

        for rows, cols in ((2, 5), (5, 2), (7, 11), (13, 8)):
            emb_i = rng.normal(size=(rows, 16)).astype(np.float32)
            emb_j = rng.normal(size=(cols, 16)).astype(np.float32)

            t_i = torch.nn.functional.normalize(torch.as_tensor(emb_i), dim=-1)
            t_j = torch.nn.functional.normalize(torch.as_tensor(emb_j), dim=-1)
            cosine = torch.mm(t_i, t_j.T).clamp(-1.0, 1.0)
            similarity = torch.exp(-(1.0 - cosine))
            row_mean = similarity.mean(dim=1, keepdim=True)
            row_std = similarity.std(
                dim=1,
                keepdim=True,
                correction=0,
            )
            col_mean = similarity.mean(dim=0, keepdim=True)
            col_std = similarity.std(
                dim=0,
                keepdim=True,
                correction=0,
            )
            legacy = (
                (similarity - row_mean) / (row_std + 1e-8)
                + (similarity - col_mean) / (col_std + 1e-8)
            ) / 2.0
            legacy = legacy.numpy().astype(np.float32, copy=False)

            optimized = similarity_matrix.compute_score_matrix_torch(
                emb_i,
                emb_j,
                device,
            )
            np.testing.assert_allclose(
                optimized,
                legacy,
                rtol=1e-5,
                atol=1e-5,
            )

            legacy_alignment = similarity_matrix.global_local_scores(
                legacy,
                similarity_matrix.GLOBAL_GAP_P,
                similarity_matrix.LOCAL_GAP_P,
            )
            optimized_alignment = similarity_matrix.global_local_scores(
                optimized,
                similarity_matrix.GLOBAL_GAP_P,
                similarity_matrix.LOCAL_GAP_P,
            )
            np.testing.assert_allclose(
                [optimized_alignment[0], optimized_alignment[2]],
                [legacy_alignment[0], legacy_alignment[2]],
                rtol=1e-5,
                atol=1e-3,
            )
            self.assertEqual(
                (int(optimized_alignment[1]), int(optimized_alignment[3])),
                (int(legacy_alignment[1]), int(legacy_alignment[3])),
            )

    def test_population_statistics_support_singleton_dimensions(self):
        singleton = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        multiple = np.array(
            [
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        device = torch.device("cpu")

        for emb_i, emb_j in (
            (singleton, multiple),
            (multiple, singleton),
            (singleton, singleton),
        ):
            score_matrix = similarity_matrix.compute_score_matrix_torch(
                emb_i,
                emb_j,
                device,
            )
            self.assertTrue(np.isfinite(score_matrix).all())

        np.testing.assert_array_equal(
            similarity_matrix.compute_score_matrix_torch(
                singleton,
                singleton,
                device,
            ),
            np.zeros((1, 1), dtype=np.float32),
        )

    def test_unsupported_std_mean_falls_back_once_per_device(self):
        emb_i = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
        emb_j = np.arange(8, dtype=np.float32).reshape(2, 4) + 2.0

        with mock.patch.object(
            torch,
            "std_mean",
            side_effect=RuntimeError("backend operation is unsupported"),
        ) as std_mean:
            first = similarity_matrix.compute_score_matrix_torch(
                emb_i,
                emb_j,
                torch.device("cpu"),
            )
            second = similarity_matrix.compute_score_matrix_torch(
                emb_i,
                emb_j,
                torch.device("cpu"),
            )

        self.assertEqual(std_mean.call_count, 1)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(
            similarity_matrix.std_mean_support_cache[("cpu", None)]
        )

    def test_backend_stream_routing_is_portable(self):
        class Device:
            def __init__(self, device_type):
                self.type = device_type

            def __str__(self):
                return self.type

        # PyTorch exposes AMD ROCm devices through the CUDA device API.
        self.assertTrue(
            similarity_matrix._supports_explicit_streams(Device("cuda"))
        )
        self.assertTrue(
            similarity_matrix._supports_explicit_streams(Device("xpu"))
        )
        for device_type in ("mps", "privateuseone", "cpu"):
            self.assertFalse(
                similarity_matrix._supports_explicit_streams(
                    Device(device_type)
                )
            )

    def test_cuda_stream_count_is_bounded_independently_from_cpu_workers(self):
        with mock.patch.object(similarity_matrix, "GPU_STREAMS", 4):
            self.assertEqual(
                similarity_matrix._accelerator_worker_count(
                    torch.device("cuda"),
                    16,
                ),
                4,
            )
            self.assertEqual(
                similarity_matrix._accelerator_worker_count(
                    torch.device("mps"),
                    16,
                ),
                1,
            )

    def test_alignment_kernels_release_the_gil_for_thread_parallelism(self):
        self.assertTrue(
            similarity_matrix.global_local_scores.targetoptions["nogil"]
        )

    def test_process_batch_routes_accelerators_to_single_producer_pipeline(self):
        expected = [(0, 1, 1.0, 1, 2.0, 1)]
        tasks = [(0, 1, "a", "b")]

        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(
                    similarity_matrix,
                    "RESULTS_DIR",
                    temp_dir,
                ), mock.patch.object(
                    similarity_matrix.Hardware_Utils,
                    "get_optimal_device",
                    return_value=torch.device("cuda"),
                ) as get_device, mock.patch.object(
                    similarity_matrix,
                    "process_accelerated_tasks",
                    return_value=expected,
                ) as accelerated, mock.patch.object(
                    similarity_matrix,
                    "process_cpu_tasks",
                ) as cpu:
            similarity_matrix.process_batch(
                tasks,
                batch_id=5,
                workers=16,
                input_h5="unused.h5",
                embedding_checksum="checksum",
            )

            output_path = os.path.join(temp_dir, "batch_00005.h5")
            with h5py.File(output_path, "r") as hf:
                self.assertEqual(hf.attrs["embedding_checksum"], "checksum")
                np.testing.assert_array_equal(hf["i"][:], [0])
                np.testing.assert_array_equal(hf["j"][:], [1])
                self.assertNotIn("paths", hf)

        get_device.assert_called_once_with()
        accelerated.assert_called_once()
        self.assertEqual(accelerated.call_args.args[1], 16)
        cpu.assert_not_called()

    def test_process_batch_keeps_parallel_whole_pair_cpu_fallback(self):
        expected = [(0, 1, 1.0, 1, 2.0, 1)]
        tasks = [(0, 1, "a", "b")]

        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(
                    similarity_matrix,
                    "RESULTS_DIR",
                    temp_dir,
                ), mock.patch.object(
                    similarity_matrix.Hardware_Utils,
                    "get_optimal_device",
                    return_value=torch.device("cpu"),
                ), mock.patch.object(
                    similarity_matrix,
                    "process_accelerated_tasks",
                ) as accelerated, mock.patch.object(
                    similarity_matrix,
                    "process_cpu_tasks",
                    return_value=expected,
                ) as cpu:
            similarity_matrix.process_batch(
                tasks,
                batch_id=6,
                workers=8,
                input_h5="unused.h5",
                embedding_checksum="checksum",
            )

        accelerated.assert_not_called()
        cpu.assert_called_once_with(tasks, 8, "unused.h5", 6)

    def test_accelerator_tuning_precedes_overall_progress_bar(self):
        events = []

        class ProgressRecorder:
            def __init__(self, *args, **kwargs):
                events.append(("overall", kwargs.get("desc")))

            def update(self, amount):
                events.append(("update", amount))

            def close(self):
                events.append(("close", None))

        manifest = mock.Mock(
            model_name="test-model",
            saving_mode="float32",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_h5 = os.path.join(temp_dir, "embeddings.h5")
            with h5py.File(input_h5, "w"):
                pass

            with mock.patch.object(
                similarity_matrix,
                "FULL_INPUT_HDF5",
                input_h5,
            ), mock.patch.object(
                similarity_matrix,
                "FINAL_OUTPUT_NET",
                os.path.join(temp_dir, "network.h5"),
            ), mock.patch.object(
                similarity_matrix,
                "RESULTS_DIR",
                os.path.join(temp_dir, "batches"),
            ), mock.patch.object(
                similarity_matrix,
                "BATCH_SIZE",
                2,
            ), mock.patch.object(
                similarity_matrix,
                "load_embedding_metadata",
                return_value=(
                    ["a", "b", "c"],
                    ["a", "b", "c"],
                    [2, 2, 2],
                    manifest,
                ),
            ), mock.patch.object(
                similarity_matrix,
                "calculate_file_hash",
                return_value="checksum",
            ), mock.patch.object(
                similarity_matrix,
                "scan_existing_batches",
            ), mock.patch.object(
                similarity_matrix,
                "_benchmark_processing_plans",
                side_effect=lambda *args, **kwargs: events.append(
                    ("benchmark", len(args[0]))
                ) or [
                    similarity_matrix.Hardware_Utils.BenchmarkResult(
                        similarity_matrix.Hardware_Utils.DeviceCandidate(
                            "cuda:0",
                            "Test CUDA",
                            torch.device("cuda:0"),
                            "cuda",
                            0,
                            True,
                        ),
                        1.0,
                        lanes=3,
                    )
                ],
            ), mock.patch.object(
                similarity_matrix,
                "tqdm",
                ProgressRecorder,
            ), mock.patch.object(
                similarity_matrix,
                "process_batch",
                side_effect=lambda *args, **kwargs: events.append(
                    ("batch", kwargs["accelerator_workers"])
                ),
            ), mock.patch.object(
                similarity_matrix,
                "compile_final_output",
            ), mock.patch.object(
                similarity_matrix,
                "set_start_method",
            ), mock.patch.object(
                similarity_matrix,
                "configure_runtime_paths",
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                similarity_matrix.run_job_distributor()

        self.assertEqual(
            events[:3],
            [
                ("benchmark", 3),
                ("overall", "Overall Progress"),
                ("batch", 3),
            ],
        )

    def test_pending_pair_sample_is_deterministic_and_cost_stratified(self):
        headers = ["a", "b", "c", "d"]
        lengths = [1, 2, 4, 8]
        computed = np.zeros((4, 4), dtype=bool)
        first = similarity_matrix._representative_pending_pairs(
            headers, lengths, computed, None, 6, 3
        )
        second = similarity_matrix._representative_pending_pairs(
            headers, lengths, computed, None, 6, 3
        )
        self.assertEqual(first, second)
        costs = [lengths[row] * lengths[column] for row, column, _, _ in first]
        self.assertEqual(costs[0], 2)
        self.assertEqual(costs[-1], 32)


if __name__ == "__main__":
    unittest.main()
