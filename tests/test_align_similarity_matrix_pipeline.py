"""Tests of Align_Similarity_Matrix: input paths and embedding metadata,
score matrices and alignment workers, CPU/scalar pipelines with their
first-batch benchmark trials, plan and precision selection, prefiltering,
batch routing, and startup/main() error handling.

Tiled/GPU batch execution tests live in test_align_similarity_matrix_tiled.py;
resume scans, partial-batch checkpoints, and network compilation live in
test_align_similarity_matrix_resume.py; engine-only tests live in
test_alignment_engine.py.
"""
import io
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import torch


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
    from utilities import HDF5_Storage as Embedding_HDF5

from tests.align_pipeline_fixtures import run_align_main  # noqa: E402
from tests.alignment_fixtures import ImmediateExecutor, RecordingTrial  # noqa: E402
from tests.embedding_fixtures import write_embedding_arrays  # noqa: E402


class AlignmentPipelineTests(unittest.TestCase):
    def setUp(self):
        # Each test starts with an empty capability cache, and an entry it
        # records (the std_mean fallback test stores "unsupported" for the
        # CPU) does not reach the score matrices of later tests and modules.
        patcher = mock.patch.object(similarity_matrix, "std_mean_support_cache", {})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_runtime_path_configuration_preserves_none_as_unselected(self):
        with mock.patch.object(similarity_matrix, "INPUT_HDF5", None):
            with self.assertRaisesRegex(ValueError, "No embeddings file"):
                similarity_matrix.configure_runtime_paths()

    def test_cpu_benchmark_drains_warmup_and_reuses_one_pool(self):
        events = []

        class FakePool:
            instances = 0

            def __init__(self, processes, initializer, initargs):
                type(self).instances += 1
                events.append("pool-open")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                events.append("pool-close")
                return False

            def map_async(self, function, chunk, chunksize):
                def get():
                    events.extend(f"pair-{task[1]}" for task in chunk)
                    return [function(task) for task in chunk]

                return SimpleNamespace(get=get)

        tasks = [(0, column, "a", f"h{column}") for column in range(1, 5)]
        trial = RecordingTrial(events)
        with mock.patch.object(
            similarity_matrix, "Pool", FakePool
        ), mock.patch.object(
            similarity_matrix,
            "calculate_cpu_pair",
            side_effect=lambda task: (task[0], task[1], 1.0, 1, 2.0, 1),
        ):
            results = similarity_matrix.process_cpu_tasks(
                tasks,
                workers=2,
                input_h5="unused.h5",
                batch_id=-1,
                show_progress=False,
                benchmark_trial=trial,
            )

        # Four pairs are fewer than the warm-up minimum, so the warm-up runs all
        # of them and the timed phase starts again from the first pair.
        pairs = [f"pair-{column}" for column in range(1, 5)]
        self.assertEqual(FakePool.instances, 1)
        self.assertEqual(
            events,
            ["pool-open", *pairs, "trial-start", *pairs, "trial-stop", "pool-close"],
        )
        # Only the timed phase's results are returned.
        self.assertEqual(
            sorted(results),
            [(0, column, 1.0, 1, 2.0, 1) for column in range(1, 5)],
        )
        self.assertEqual((trial.submitted, trial.completed), (4, 4))

    def test_cpu_pool_uses_explicit_spawn_context(self):
        # Linux runs must never fork while torch OpenMP threads, CUDA, or the
        # partial-RAM HDF5 prefetch thread hold internal locks.
        self.assertEqual(similarity_matrix._POOL_CONTEXT.get_start_method(), "spawn")
        self.assertIs(similarity_matrix.Pool.__self__, similarity_matrix._POOL_CONTEXT)

    def test_scalar_benchmark_keeps_executors_alive_across_phase_boundary(self):
        events = []
        ImmediateExecutor.instances.clear()

        def score(args):
            row, column, _left, _right, _device = args
            events.append(f"score-{column}")
            return row, column, np.zeros((1, 1), dtype=np.float32)

        def align(args):
            row, column, _matrix = args
            events.append(f"align-{column}")
            return row, column, 1.0, 1, 2.0, 1

        tasks = [(0, column, "a", f"h{column}") for column in range(1, 5)]
        trial = RecordingTrial(events)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_h5 = os.path.join(temp_dir, "embeddings.h5")
            write_embedding_arrays(
                input_h5,
                {
                    "a": np.ones((1, 2), dtype=np.float32),
                    **{f"h{column}": np.ones((1, 2), dtype=np.float32) for column in range(1, 5)},
                },
                saving_mode="float32",
            )

            real_h5_file = h5py.File
            with mock.patch.object(
                similarity_matrix, "ThreadPoolExecutor", ImmediateExecutor
            ), mock.patch.object(
                similarity_matrix,
                "_compute_accelerated_matrix",
                side_effect=score,
            ), mock.patch.object(
                similarity_matrix,
                "calculate_alignment_data",
                side_effect=align,
            ), mock.patch.object(
                similarity_matrix.h5py,
                "File",
                side_effect=real_h5_file,
            ) as opened:
                results = similarity_matrix._run_scalar_accelerated_pipeline(
                    tasks,
                    workers=2,
                    input_h5=input_h5,
                    device=torch.device("cpu"),
                    batch_id=-1,
                    accelerator_workers=1,
                    show_progress=False,
                    benchmark_trial=trial,
                )

        # The warm-up covers all four pairs (fewer than the warm-up minimum);
        # the timed phase then starts again from the first pair.
        trial_start = events.index("trial-start")
        trial_stop = events.index("trial-stop")
        pair_events = [
            f"{stage}-{column}" for column in range(1, 5) for stage in ("score", "align")
        ]
        self.assertEqual(len(results), 4)
        self.assertEqual(len(ImmediateExecutor.instances), 2)
        self.assertEqual(opened.call_count, 1)
        self.assertCountEqual(events[:trial_start], pair_events)
        self.assertCountEqual(events[trial_start + 1:trial_stop], pair_events)
        self.assertEqual(trial_stop, len(events) - 1)
        self.assertEqual((trial.submitted, trial.completed), (4, 4))

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
            write_embedding_arrays(
                h5_path,
                {
                    "second_header": np.ones((2, 4), dtype=np.float32),
                    "first": np.ones((3, 4), dtype=np.float32),
                },
                sequences=["AC", "ACD"],
                model_name="test-model",
                saving_mode="float32",
            )

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
            write_embedding_arrays(
                h5_path,
                {
                    "a": np.ones((2, 4), dtype=np.float32),
                    "b": np.ones((3, 4), dtype=np.float32),
                    "c": np.ones((2, 4), dtype=np.float32),
                },
                saving_mode="float32",
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
                self.assertEqual(hf.attrs["matmul_precision"], "ieee_fp32")
                np.testing.assert_array_equal(hf["i"][:], [0])
                np.testing.assert_array_equal(hf["j"][:], [1])
                self.assertNotIn("paths", hf)
            self.assertFalse(os.path.exists(output_path + ".partial"))

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
        cpu.assert_called_once()
        self.assertEqual(cpu.call_args.args, (tasks, 8, "unused.h5", 6))
        self.assertIsInstance(
            cpu.call_args.kwargs["result_callback"],
            similarity_matrix._PartialBatchWriter,
        )

    def test_accelerator_tuning_precedes_overall_progress_bar(self):
        events = []

        class ProgressRecorder:
            def __init__(self, iterable=None, *args, **kwargs):
                self.iterable = iterable
                if kwargs.get("desc") == "Overall Progress":
                    events.append(("overall", kwargs.get("desc")))

            def __iter__(self):
                return iter(self.iterable or ())

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
            write_embedding_arrays(
                input_h5,
                {header: np.ones((2, 2), dtype=np.float32) for header in ("a", "b", "c")},
                saving_mode="float32",
            )

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
                "EmbeddingTileStore",
                return_value=mock.Mock(
                    fully_cached=False,
                    cached_bytes=0,
                ),
            ), mock.patch.object(
                similarity_matrix,
                "_resolve_active_matmul_precision",
                return_value="ieee_fp32",
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
                ("benchmark", 2),
                ("overall", "Overall Progress"),
                ("batch", 3),
            ],
        )

    def test_auto_precision_benchmarks_all_four_plan_combinations(self):
        cuda = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cuda:0",
            "Test CUDA",
            torch.device("cuda:0"),
            "cuda",
            0,
            True,
        )
        tasks = [
            (0, 1, "a", "b"),
            (0, 2, "a", "c"),
            (1, 2, "b", "c"),
        ]
        results = [
            (0, 1, 1.0, 2, 2.0, 2),
            (0, 2, 1.5, 2, 2.5, 2),
            (1, 2, 1.2, 2, 2.2, 2),
        ]
        store = mock.Mock(path="unused.h5")
        clock = iter([0.0, 1.0, 1.0, 1.4, 2.0, 3.0, 3.0, 3.4])
        memory_plan = mock.Mock(free_bytes=12 << 30, total_bytes=16 << 30)
        safe_estimate = mock.Mock(
            feasible=True,
            projected_peak_bytes=10 << 30,
            safe_peak_bytes=13 << 30,
            tile_bytes=1 << 30,
            transient_bytes=2 << 30,
            per_microbatch_bytes=512 << 20,
            reason="within reserved-VRAM boundary",
        )

        # The simulated CUDA device must count as NVIDIA on any host; the real
        # check reads the installed PyTorch build (CPU, ROCm and MPS say no).
        with mock.patch.object(
            similarity_matrix, "is_nvidia_cuda",
            side_effect=lambda device: device.type == "cuda",
        ), mock.patch.object(
            similarity_matrix.Hardware_Utils,
            "get_available_devices",
            return_value=[cuda],
        ), mock.patch.object(
            similarity_matrix.Hardware_Utils,
            "release_device_cache",
        ), mock.patch.object(
            similarity_matrix,
            "tiled_accelerator_support",
            return_value=(True, "mock tiled support"),
        ), mock.patch.object(
            similarity_matrix,
            "cuda_memory_plan",
            return_value=memory_plan,
        ), mock.patch.object(
            similarity_matrix,
            "estimate_cuda_working_set",
            return_value=safe_estimate,
        ), mock.patch.object(
            similarity_matrix,
            "_run_accelerated_pipeline",
            return_value=results,
        ) as scalar, mock.patch.object(
            similarity_matrix,
            "run_tiled_cuda_pipeline",
            return_value=results,
        ) as tiled, mock.patch.object(
            similarity_matrix.time,
            "perf_counter",
            side_effect=lambda: next(clock),
        ), redirect_stdout(io.StringIO()):
            precision = similarity_matrix._resolve_active_matmul_precision(
                "auto",
                None,
                tasks,
                workers=2,
                store=store,
                sequence_lengths=[2, 2, 2],
            )

        self.assertEqual(precision, "tf32")
        self.assertEqual(scalar.call_count, 3)
        self.assertEqual(tiled.call_count, 3)
        self.assertEqual(
            [len(call.args[0]) for call in scalar.call_args_list],
            [2, 3, 3],
        )
        self.assertEqual(
            [len(call.args[0]) for call in tiled.call_args_list],
            [2, 3, 3],
        )
        self.assertEqual(
            [call.kwargs["matmul_precision"] for call in scalar.call_args_list],
            ["float32", "float32", "tf32"],
        )
        self.assertEqual(
            [call.kwargs["precision"] for call in tiled.call_args_list],
            ["float32", "float32", "tf32"],
        )

    def test_forced_scalar_precision_validation_never_runs_tiled_plan(self):
        cuda = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cuda:0",
            "Test CUDA",
            torch.device("cuda:0"),
            "cuda",
            0,
            True,
        )
        tasks = [(0, 1, "a", "b"), (0, 2, "a", "c")]
        results = [(0, 1, 1.0, 2, 2.0, 2), (0, 2, 1.5, 2, 2.5, 2)]
        safe_estimate = mock.Mock(
            feasible=True,
            projected_peak_bytes=10 << 30,
            safe_peak_bytes=13 << 30,
            tile_bytes=0,
            transient_bytes=2 << 30,
            per_microbatch_bytes=512 << 20,
            reason="within reserved-VRAM boundary",
        )
        memory_plan = mock.Mock(free_bytes=12 << 30, total_bytes=16 << 30)
        clock = iter([0.0, 1.0, 1.0, 1.4])
        with mock.patch.object(
            similarity_matrix, "is_nvidia_cuda",
            side_effect=lambda device: device.type == "cuda",
        ), mock.patch.object(
            similarity_matrix, "EXECUTION_MODE", "scalar"
        ), mock.patch.object(
            similarity_matrix.Hardware_Utils,
            "get_available_devices",
            return_value=[cuda],
        ), mock.patch.object(
            similarity_matrix.Hardware_Utils, "release_device_cache"
        ), mock.patch.object(
            similarity_matrix, "cuda_memory_plan", return_value=memory_plan
        ), mock.patch.object(
            similarity_matrix,
            "estimate_cuda_working_set",
            return_value=safe_estimate,
        ), mock.patch.object(
            similarity_matrix,
            "_run_accelerated_pipeline",
            return_value=results,
        ) as scalar, mock.patch.object(
            similarity_matrix, "run_tiled_cuda_pipeline"
        ) as tiled, mock.patch.object(
            similarity_matrix.time,
            "perf_counter",
            side_effect=lambda: next(clock),
        ), redirect_stdout(io.StringIO()):
            precision = similarity_matrix._resolve_active_matmul_precision(
                "auto",
                None,
                tasks,
                workers=2,
                store=mock.Mock(path="unused.h5"),
                sequence_lengths=[2, 2, 2],
            )

        self.assertEqual(precision, "tf32")
        self.assertEqual(scalar.call_count, 3)
        tiled.assert_not_called()

    def test_first_batch_trials_run_each_feasible_configuration_once(self):
        cpu = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        cuda = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cuda:0",
            "Test CUDA",
            torch.device("cuda:0"),
            "cuda",
            0,
            True,
        )
        tasks = [(0, column, "a", f"h{column}") for column in range(1, 7)]
        output = io.StringIO()
        memory_plan = mock.Mock(free_bytes=12 << 30, total_bytes=16 << 30)
        safe_estimate = mock.Mock(
            feasible=True,
            projected_peak_bytes=10 << 30,
            safe_peak_bytes=13 << 30,
            tile_bytes=1 << 30,
            transient_bytes=2 << 30,
            per_microbatch_bytes=512 << 20,
            reason="within reserved-VRAM boundary",
        )

        def complete_benchmark(*_args, **kwargs):
            trial = kwargs["benchmark_trial"]
            trial.start()
            trial.submitted = trial.completed = 6
            trial.stop(6)
            return []

        with mock.patch.object(
            similarity_matrix, "DEVICE_SELECTION", "auto"
        ), mock.patch.object(
            similarity_matrix.Hardware_Utils,
            "get_available_devices",
            return_value=[cpu, cuda],
        ), mock.patch.object(
            similarity_matrix.Hardware_Utils,
            "release_device_cache",
        ), mock.patch.object(
            similarity_matrix,
            "tiled_accelerator_support",
            return_value=(True, "mock tiled support"),
        ), mock.patch.object(
            similarity_matrix,
            "cuda_memory_plan",
            return_value=memory_plan,
        ), mock.patch.object(
            similarity_matrix,
            "estimate_cuda_working_set",
            return_value=safe_estimate,
        ), mock.patch.object(
            similarity_matrix,
            "_accelerator_lane_candidates",
            return_value=[1, 2],
        ), mock.patch.object(
            similarity_matrix,
            "process_cpu_tasks",
            side_effect=complete_benchmark,
        ) as cpu_pipeline, mock.patch.object(
            similarity_matrix,
            "_run_accelerated_pipeline",
            side_effect=complete_benchmark,
        ) as scalar_pipeline, mock.patch.object(
            similarity_matrix,
            "run_tiled_accelerator_pipeline",
            side_effect=complete_benchmark,
        ) as tiled_pipeline, redirect_stdout(output):
            plans = similarity_matrix._benchmark_processing_plans(
                tasks,
                workers=2,
                input_h5="unused.h5",
                batch_id=0,
                embedding_store=mock.Mock(),
                sequence_lengths=[2] * 7,
                matmul_precision="ieee_fp32",
            )

        self.assertTrue(plans)
        self.assertEqual(
            [len(call.args[0]) for call in cpu_pipeline.call_args_list],
            [6],
        )
        self.assertEqual(
            [len(call.args[0]) for call in scalar_pipeline.call_args_list],
            [6, 6],
        )
        self.assertEqual(
            [len(call.args[0]) for call in tiled_pipeline.call_args_list],
            [6, 6],
        )
        for call in scalar_pipeline.call_args_list + tiled_pipeline.call_args_list:
            self.assertEqual(call.args[0], tasks)
            self.assertIn("benchmark_trial", call.kwargs)

    def test_a_failed_gpu_trial_shows_the_free_memory_and_leaves_a_note_for_an_out_of_memory_error(self):
        cuda = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cuda:0", "Test CUDA", torch.device("cuda:0"), "cuda", 0, True
        )
        tasks = [(0, column, "a", f"h{column}") for column in range(1, 7)]
        memory_plan = mock.Mock(free_bytes=12 << 30, total_bytes=16 << 30)
        estimate = mock.Mock(
            feasible=True, projected_peak_bytes=7 << 30, safe_peak_bytes=10 << 30,
            reason="within reserved-VRAM boundary",
        )
        output = io.StringIO()
        hardware = similarity_matrix.Hardware_Utils

        def fail(*_args, **_kwargs):
            raise RuntimeError(
                "CUDA error: out of memory\nSearch for `cudaErrorMemoryAllocation' for more information.\n"
                "Sticky error detected\n"
            )

        system = {"commit_used": int(111.2 * (1 << 30)), "commit_limit": int(111.8 * (1 << 30)),
                  "ram_available": int(6.2 * (1 << 30))}
        with mock.patch.object(hardware, "read_system_memory", return_value=system), \
                mock.patch.object(similarity_matrix, "DEVICE_SELECTION", "cuda:0"), \
                mock.patch.object(similarity_matrix, "EXECUTION_MODE", "scalar"), \
                mock.patch.object(hardware, "get_available_devices", return_value=[cuda]), \
                mock.patch.object(hardware, "release_device_cache"), \
                mock.patch.object(hardware, "begin_gpu_trial", return_value=(int(11.2 * (1 << 30)), 12 << 30)), \
                mock.patch.object(hardware.Tool_Supervisor, "record_note") as note, \
                mock.patch.object(similarity_matrix, "tiled_accelerator_support", return_value=(True, "mock")), \
                mock.patch.object(similarity_matrix, "cuda_memory_plan", return_value=memory_plan), \
                mock.patch.object(similarity_matrix, "estimate_cuda_working_set", return_value=estimate), \
                mock.patch.object(similarity_matrix, "_accelerator_lane_candidates", return_value=[1]), \
                mock.patch.object(similarity_matrix, "_run_accelerated_pipeline", side_effect=fail), \
                redirect_stdout(output):
            with self.assertRaises(RuntimeError):
                similarity_matrix._benchmark_processing_plans(
                    tasks, workers=2, input_h5="unused.h5", batch_id=0,
                    embedding_store=mock.Mock(), sequence_lengths=[2] * 7, matmul_precision="ieee_fp32",
                )

        lines = output.getvalue().splitlines()
        row = next(line for line in lines if "CUDA error: out of memory" in line)
        # The memory text sits on the row; the error's other lines follow, indented.
        self.assertTrue(row.endswith(
            "RuntimeError: CUDA error: out of memory (free before the test: 11.2 of 12.0 GB; "
            "Windows virtual memory: 111.2 of 111.8 GB in use; RAM available: 6.2 GB)"), row)
        after = lines[lines.index(row) + 1:]
        self.assertEqual(after[:2], ["    Search for `cudaErrorMemoryAllocation' for more information.",
                                     "    Sticky error detected"])
        note.assert_called_once()
        self.assertEqual(note.call_args.kwargs["free_bytes"], int(11.2 * (1 << 30)))

    def test_prefilter_requires_exactly_the_kept_pairs(self):
        n = 90
        rng = np.random.default_rng(5)
        lengths = rng.integers(40, 160, n)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            names = [f"s{index}" for index in range(n)]
            write_embedding_arrays(
                path,
                {
                    name: rng.standard_normal((int(length), 6)).astype(np.float16)
                    for name, length in zip(names, lengths)
                },
                saving_mode="float16",
            )
            settings = {
                "FULL_INPUT_HDF5": path,
                "PREFILTER_STRENGTH": 40,
            }
            with mock.patch.multiple(similarity_matrix, **settings), \
                    redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
                required_keys, cut = similarity_matrix.prefilter_pair_keys(names, list(lengths))
            with h5py.File(path, "r") as hf:
                pooled = np.array([np.max(hf["embeddings"][name][:], axis=0) for name in names],
                                  dtype=np.float32)
        similarity = similarity_matrix.Prefilter_Calibration.length_adjusted_similarity(
            pooled, lengths, 2.0
        )
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        threshold = np.percentile(similarity[upper], 40)
        kept = (similarity >= threshold) & upper
        rr, cc = np.nonzero(kept)
        want_keys = rr.astype(np.int64) * n + cc.astype(np.int64)
        self.assertEqual(cut.threshold, threshold)
        self.assertEqual(cut.kept, int(kept.sum()))
        self.assertEqual(cut.skipped, int(upper.sum() - kept.sum()))
        # Exactly the kept pairs are required: no calibration sample is added among the skipped ones.
        np.testing.assert_array_equal(required_keys, want_keys)
        text = stdout.getvalue()
        self.assertIn(f"Adjusted Similarity Threshold: {threshold:.6f}", text)
        self.assertIn(f"Retaining {cut.kept} / {int(upper.sum())} pairs", text)
        self.assertNotIn("Calibration sample", text)

    def test_prefilter_uses_the_shared_pooling_and_power_and_requires_only_the_kept_pairs(self):
        n = 120
        rng = np.random.default_rng(6)
        lengths = rng.integers(50, 60, n)
        names = [f"s{index}" for index in range(n)]
        embeddings = {
            name: rng.standard_normal((int(length), 4)).astype(np.float32)
            for name, length in zip(names, lengths)
        }

        class FakeFile(dict):
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        with mock.patch.multiple(similarity_matrix, PREFILTER_STRENGTH=50), \
                mock.patch.multiple(similarity_matrix.Prefilter_Calibration, POOLING="mean", LENGTH_RATIO_POWER=1.0), \
                mock.patch.object(similarity_matrix.h5py, "File",
                                  return_value=FakeFile(embeddings=embeddings)), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            required_keys, cut = similarity_matrix.prefilter_pair_keys(names, list(lengths))
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        similarity = similarity_matrix.Prefilter_Calibration.length_adjusted_similarity(
            np.array([embeddings[name].mean(axis=0) for name in names], dtype=np.float32), lengths, 1.0)
        threshold = float(np.percentile(similarity[upper], 50))
        rr, cc = np.nonzero((similarity >= threshold) & upper)
        want_keys = rr.astype(np.int64) * n + cc.astype(np.int64)
        self.assertEqual(cut.threshold, threshold)
        self.assertEqual(int(required_keys.size), cut.kept)
        np.testing.assert_array_equal(required_keys, want_keys)
        self.assertTrue(np.all((required_keys // n) < (required_keys % n)))

    def test_main_exits_nonzero_when_the_job_cannot_start(self):
        # The MCP job runner reports exit code 0 as "succeeded", so these used
        # to reach agents as successful jobs that wrote nothing.
        cases = (
            ("no input", {"INPUT_HDF5": None}, None,
             "❌ Cannot start alignment:\nNo embeddings file has been selected."),
            ("bad precision", {"ACCELERATOR_PRECISION": "fp8"}, None,
             "❌ Cannot start alignment:\nACCELERATOR_PRECISION must be"),
            ("missing input", {}, None, "Embeddings file not found."),
            ("invalid input", {}, b"not an embeddings file",
             "❌ Cannot start alignment:\nEmbedding file '"),
        )
        for label, patches, input_bytes, message in cases:
            with self.subTest(label), tempfile.TemporaryDirectory() as temp_dir:
                if input_bytes is not None:
                    with open(
                        os.path.join(temp_dir, "set_[model]_embeddings.h5"), "wb"
                    ) as handle:
                        handle.write(input_bytes)

                exit_code, stdout, compile_output = run_align_main(
                    temp_dir, **patches
                )

                self.assertEqual(exit_code, 1)
                self.assertIn(message, stdout)
                compile_output.assert_not_called()

    def test_main_exits_nonzero_when_adaptive_setup_is_rejected(self):
        manifest = mock.Mock(model_name="model", saving_mode="float32")
        cases = (
            ("host cache", {
                "EmbeddingTileStore": mock.Mock(
                    side_effect=ValueError("HOST_CACHE_GB is invalid.")
                ),
            }, 1, "❌ Invalid adaptive alignment configuration:\nHOST_CACHE_GB"),
            ("precision", {
                "_resolve_active_matmul_precision": mock.Mock(
                    side_effect=ValueError("bf16 failed validation.")
                ),
            }, 1, "❌ Invalid precision configuration:\nbf16 failed"),
            ("control", {}, 0, "Pairs queued for calculation: 1"),
        )
        for label, failure, expected_code, message in cases:
            with self.subTest(label), tempfile.TemporaryDirectory() as temp_dir:
                with open(
                    os.path.join(temp_dir, "set_[model]_embeddings.h5"), "wb"
                ) as handle:
                    handle.write(b"metadata is mocked")
                run_batch = mock.Mock(return_value=0)
                patches = {
                    "load_embedding_metadata": mock.Mock(return_value=(
                        ["a", "b"], ["a", "b"], [2, 2], manifest,
                    )),
                    "scan_existing_batches": mock.Mock(return_value=None),
                    "Hardware_Utils": mock.Mock(**{
                        "resolve_device_selection.return_value":
                            mock.Mock(is_cpu=True),
                    }),
                    "EmbeddingTileStore": mock.Mock(return_value=mock.Mock(
                        fully_cached=False, cached_bytes=0,
                    )),
                    "_resolve_active_matmul_precision": mock.Mock(
                        return_value="ieee_fp32"
                    ),
                    "_benchmark_processing_plans": mock.Mock(return_value=[]),
                    "_run_batch_with_ranked_plans": run_batch,
                    "tqdm": mock.MagicMock(),
                    **failure,
                }

                exit_code, stdout, compile_output = run_align_main(
                    temp_dir, **patches
                )

                self.assertEqual(exit_code, expected_code)
                self.assertIn(message, stdout)
                self.assertEqual(run_batch.call_count, 1 - expected_code)
                self.assertEqual(compile_output.call_count, 1 - expected_code)


if __name__ == "__main__":
    unittest.main()
