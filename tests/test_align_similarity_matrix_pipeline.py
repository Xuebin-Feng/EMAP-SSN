import io
import os
import sys
import tempfile
import threading
import unittest
from collections import Counter
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

with mock.patch.dict(os.environ, {
    "SSN_TOOL_SETTINGS_SCRIPT": "Align_Similarity_Matrix.py",
    "SSN_TOOL_SETTINGS_FILE": os.path.join(PROJECT_ROOT, "tests", "nonexistent-settings.json"),
}), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as similarity_matrix
    import Embedding_HDF5
    from utilities import Embedding_Alignment_Engine as alignment_engine


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

    def test_execution_mode_normalization_accepts_only_supported_values(self):
        self.assertEqual(alignment_engine.normalize_execution_mode(None), "auto")
        self.assertEqual(
            alignment_engine.normalize_execution_mode(" TILED "), "tiled"
        )
        with self.assertRaisesRegex(ValueError, "auto.*scalar.*tiled"):
            alignment_engine.normalize_execution_mode("batched")

    def test_tiled_progress_advances_for_any_completed_future(self):
        earlier = Future()
        later = Future()
        later.set_result((4, 9, 1.0, 2, 3.0, 4))
        pending = {earlier, later}
        results = []
        progress = mock.Mock()

        completed = alignment_engine._drain_completed_alignment_futures(
            pending,
            results,
            progress=progress,
            block=False,
        )

        self.assertEqual(completed, 1)
        self.assertEqual(results, [(4, 9, 1.0, 2, 3.0, 4)])
        self.assertEqual(pending, {earlier})
        progress.update.assert_called_once_with(1)

    def test_cpu_benchmark_drains_warmup_and_reuses_one_pool(self):
        events = []

        class FakePool:
            instances = 0

            def __init__(self, processes, initializer, initargs):
                self.processes = processes
                self.initializer = initializer
                self.initargs = initargs
                type(self).instances += 1
                events.append("pool-open")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                events.append("pool-close")
                return False

            def imap_unordered(self, function, tasks, chunksize):
                for task in tasks:
                    events.append(f"pair-{task[1]}")
                    yield function(task)

        class Timer:
            def start(self):
                events.append("timer-start")

            def stop(self):
                events.append("timer-stop")

        tasks = [(0, column, "a", f"h{column}") for column in range(1, 5)]
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
                warmup_task_count=2,
                benchmark_timer=Timer(),
            )

        self.assertEqual(FakePool.instances, 1)
        self.assertEqual(len(results), 4)
        self.assertEqual(
            events,
            [
                "pool-open",
                "pair-1",
                "pair-2",
                "timer-start",
                "pair-3",
                "pair-4",
                "timer-stop",
                "pool-close",
            ],
        )

    def test_scalar_benchmark_keeps_executors_alive_across_phase_boundary(self):
        events = []
        ImmediateExecutor.instances.clear()

        class Timer:
            def start(self):
                events.append("timer-start")

            def stop(self):
                events.append("timer-stop")

        def score(args):
            row, column, _left, _right, _device = args
            events.append(f"score-{column}")
            return row, column, np.zeros((1, 1), dtype=np.float32)

        def align(args):
            row, column, _matrix = args
            events.append(f"align-{column}")
            return row, column, 1.0, 1, 2.0, 1

        tasks = [(0, column, "a", f"h{column}") for column in range(1, 5)]
        with tempfile.TemporaryDirectory() as temp_dir:
            input_h5 = os.path.join(temp_dir, "embeddings.h5")
            with h5py.File(input_h5, "w") as hf:
                group = hf.create_group("embeddings")
                group.create_dataset("a", data=np.ones((1, 2), dtype=np.float32))
                for column in range(1, 5):
                    group.create_dataset(
                        f"h{column}", data=np.ones((1, 2), dtype=np.float32)
                    )

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
            ):
                results = similarity_matrix._run_scalar_accelerated_pipeline(
                    tasks,
                    workers=2,
                    input_h5=input_h5,
                    device=torch.device("cpu"),
                    batch_id=-1,
                    accelerator_workers=1,
                    show_progress=False,
                    warmup_task_count=2,
                    benchmark_timer=Timer(),
                )

        timer_start = events.index("timer-start")
        timer_stop = events.index("timer-stop")
        before_timer = events[:timer_start]
        timed_region = events[timer_start + 1:timer_stop]
        self.assertEqual(len(results), 4)
        self.assertEqual(len(ImmediateExecutor.instances), 2)
        self.assertTrue(all(f"score-{column}" in before_timer for column in (1, 2)))
        self.assertTrue(all(f"align-{column}" in before_timer for column in (1, 2)))
        self.assertTrue(all(f"score-{column}" in timed_region for column in (3, 4)))
        self.assertTrue(all(f"align-{column}" in timed_region for column in (3, 4)))

    def test_tiled_benchmark_drains_midpoint_with_one_context(self):
        events = []

        class FakeEvent:
            def record(self, stream):
                return None

            def query(self):
                return True

            def synchronize(self):
                return None

        class FakeBackend:
            device_type = "cuda"

            def __init__(self):
                self.streams = []

            def supports_tiled(self, require_memory=True):
                return True, "mock backend"

            def create_stream(self):
                stream = object()
                self.streams.append(stream)
                return stream

            def stream_context(self, stream):
                return mock.MagicMock()

            def create_event(self):
                return FakeEvent()

            def empty_cache(self):
                events.append("empty-cache")

        class Timer:
            def start(self):
                events.append("timer-start")

            def stop(self):
                events.append("timer-stop")

        def score_batch(row_tensor, target_tensors, target_lengths):
            columns = max(int(length) for length in target_lengths)
            return torch.zeros(
                (len(target_tensors), int(row_tensor.shape[0]), columns),
                dtype=torch.float32,
            )

        def align(args):
            row, column, _matrix = args
            events.append(f"align-{column}")
            return row, column, 1.0, 1, 2.0, 1

        headers = ["a", "b", "c", "d", "e"]
        tasks = [(0, column, "a", headers[column]) for column in range(1, 5)]
        backend = FakeBackend()
        plan = alignment_engine.CudaMemoryPlan(
            free_bytes=1 << 30,
            total_bytes=1 << 30,
            usable_bytes=1 << 30,
            tile_cache_bytes=1 << 20,
            matrix_pool_bytes=1 << 20,
            matrix_bytes=1 << 19,
            reserve_bytes=0,
            lanes=1,
            inflight_slots=2,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            input_h5 = os.path.join(temp_dir, "embeddings.h5")
            with h5py.File(input_h5, "w") as hf:
                group = hf.create_group("embeddings")
                for header in headers:
                    group.create_dataset(
                        header, data=np.ones((1, 2), dtype=np.float32)
                    )
            store = alignment_engine.EmbeddingTileStore(input_h5, headers, 0)
            real_h5_file = h5py.File
            with mock.patch.object(
                alignment_engine,
                "get_accelerator_backend",
                return_value=backend,
            ), mock.patch.object(
                alignment_engine,
                "_to_normalized_cuda",
                side_effect=lambda array, device: torch.as_tensor(array),
            ), mock.patch.object(
                alignment_engine,
                "_batched_score_matrices",
                side_effect=score_batch,
            ), mock.patch.object(
                alignment_engine.h5py,
                "File",
                side_effect=real_h5_file,
            ) as opened:
                results = alignment_engine.run_tiled_accelerator_pipeline(
                    tasks,
                    store=store,
                    lengths=[1] * len(headers),
                    device=torch.device("cuda:0"),
                    workers=2,
                    lanes=1,
                    alignment_callback=align,
                    memory_plan_override=plan,
                    warmup_task_count=2,
                    benchmark_timer=Timer(),
                )

        timer_start = events.index("timer-start")
        timer_stop = events.index("timer-stop")
        self.assertEqual(len(results), 4)
        self.assertEqual(len(backend.streams), 1)
        self.assertEqual(opened.call_count, 1)
        self.assertTrue(
            all(f"align-{column}" in events[:timer_start] for column in (1, 2))
        )
        self.assertTrue(
            all(
                f"align-{column}" in events[timer_start + 1:timer_stop]
                for column in (3, 4)
            )
        )

    def test_mps_tiled_pipeline_uses_default_queue_without_streams(self):
        class FakeMpsBackend:
            device_type = "mps"
            supports_async_streams = False

            def __init__(self):
                self.synchronize_calls = 0
                self.empty_cache_calls = 0

            def supports_tiled(self, require_memory=True):
                return True, "mock MPS support"

            def synchronize(self):
                self.synchronize_calls += 1

            def empty_cache(self):
                self.empty_cache_calls += 1

        headers = ["a", "b", "c"]
        tasks = [(0, 1, "a", "b"), (0, 2, "a", "c")]
        backend = FakeMpsBackend()
        plan = alignment_engine.AcceleratorMemoryPlan(
            free_bytes=1 << 30,
            total_bytes=1 << 30,
            usable_bytes=1 << 30,
            tile_cache_bytes=1 << 20,
            matrix_pool_bytes=1 << 20,
            matrix_bytes=1 << 20,
            reserve_bytes=0,
            lanes=1,
            inflight_slots=1,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_h5 = os.path.join(temp_dir, "mps_embeddings.h5")
            with h5py.File(input_h5, "w") as hf:
                group = hf.create_group("embeddings")
                for header in headers:
                    group.create_dataset(
                        header, data=np.ones((2, 3), dtype=np.float32)
                    )
            store = alignment_engine.EmbeddingTileStore(input_h5, headers, 0)
            with mock.patch.object(
                alignment_engine, "get_accelerator_backend", return_value=backend
            ), mock.patch.object(
                alignment_engine,
                "_to_normalized_cuda",
                side_effect=lambda array, device: torch.as_tensor(array),
            ), mock.patch.object(
                alignment_engine,
                "_batched_score_matrices",
                side_effect=lambda row, targets, target_lengths: torch.zeros(
                    (len(targets), len(row), max(target_lengths)),
                    dtype=torch.float32,
                ),
            ):
                results = alignment_engine.run_tiled_accelerator_pipeline(
                    tasks,
                    store=store,
                    lengths=[2, 2, 2],
                    device=torch.device("mps"),
                    workers=2,
                    lanes=1,
                    alignment_callback=lambda args: (
                        args[0], args[1], 1.0, 1, 2.0, 1
                    ),
                    memory_plan_override=plan,
                )

        self.assertEqual(len(results), 2)
        self.assertGreater(backend.synchronize_calls, 0)
        self.assertGreater(backend.empty_cache_calls, 0)

    def test_restored_engine_routes_memory_planning_to_xpu_runtime(self):
        fake_xpu = mock.MagicMock()
        fake_xpu.mem_get_info.return_value = (12 << 30, 16 << 30)

        with mock.patch.object(alignment_engine.torch, "xpu", fake_xpu):
            supported, _reason = alignment_engine.tiled_accelerator_support(
                torch.device("xpu:0"), require_memory=True
            )
            plan = alignment_engine.cuda_memory_plan(
                torch.device("xpu:0"), lanes=2
            )

        self.assertTrue(supported)
        self.assertEqual(plan.free_bytes, 12 << 30)
        self.assertEqual(plan.total_bytes, 16 << 30)
        self.assertEqual(plan.lanes, 2)
        self.assertEqual(fake_xpu.mem_get_info.call_count, 2)

    def test_execution_mode_filters_candidate_variants(self):
        cpu = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        cuda = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cuda:0", "CUDA", torch.device("cuda:0"), "cuda"
        )
        xpu = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "xpu:0", "XPU", torch.device("xpu:0"), "xpu"
        )
        mps = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "mps", "MPS", torch.device("mps"), "mps"
        )
        expectations = {
            "auto": (
                ["scalar"],
                ["scalar", "tiled"],
                ["scalar", "tiled"],
                ["scalar"],
            ),
            "scalar": (
                ["scalar"], ["scalar"], ["scalar"], ["scalar"]
            ),
            "tiled": ([], ["tiled"], ["tiled"], []),
        }
        for mode, expected_variants in expectations.items():
            with self.subTest(mode=mode), mock.patch.object(
                similarity_matrix, "EXECUTION_MODE", mode
            ), mock.patch.object(
                similarity_matrix,
                "tiled_accelerator_support",
                side_effect=lambda device, require_memory=False: (
                    device.type in {"cuda", "xpu"}, "mock capability"
                ),
            ):
                for candidate, expected in zip(
                    (cpu, cuda, xpu, mps), expected_variants
                ):
                    self.assertEqual(
                        similarity_matrix._execution_variants(candidate),
                        expected,
                    )

    def test_forced_tiled_mode_rejects_missing_compatible_accelerator(self):
        cpu = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        with mock.patch.object(
            similarity_matrix, "EXECUTION_MODE", "tiled"
        ), mock.patch.object(
            similarity_matrix.Hardware_Utils,
            "get_available_devices",
            return_value=[cpu],
        ), self.assertRaisesRegex(ValueError, "CUDA/ROCm or XPU"):
            similarity_matrix._benchmark_processing_plans(
                [(0, 1, "a", "b")],
                workers=1,
                input_h5="unused.h5",
                batch_id=0,
            )

    def test_forced_tiled_mode_accepts_xpu_capability(self):
        xpu = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "xpu:0", "XPU", torch.device("xpu:0"), "xpu"
        )
        with mock.patch.object(
            similarity_matrix, "EXECUTION_MODE", "tiled"
        ), mock.patch.object(
            similarity_matrix, "DEVICE_SELECTION", "xpu:0"
        ), mock.patch.object(
            similarity_matrix.Hardware_Utils,
            "get_available_devices",
            return_value=[xpu],
        ), mock.patch.object(
            similarity_matrix,
            "tiled_accelerator_support",
            return_value=(True, "mock XPU support"),
        ):
            self.assertEqual(
                similarity_matrix._validate_execution_mode_hardware(),
                "tiled",
            )

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

    def test_batched_score_matrices_match_scalar_float32_path(self):
        rng = np.random.default_rng(20260821)
        row = rng.normal(size=(7, 12)).astype(np.float32)
        targets = [
            rng.normal(size=(length, 12)).astype(np.float32)
            for length in (3, 5, 8)
        ]
        row_tensor = torch.nn.functional.normalize(torch.as_tensor(row), dim=-1)
        target_tensors = [
            torch.nn.functional.normalize(torch.as_tensor(target), dim=-1)
            for target in targets
        ]

        actual = alignment_engine._batched_score_matrices(
            row_tensor,
            target_tensors,
            [len(target) for target in targets],
        ).numpy()

        for index, target in enumerate(targets):
            expected = similarity_matrix.compute_score_matrix_torch(
                row,
                target,
                torch.device("cpu"),
            )
            np.testing.assert_allclose(
                actual[index, :, :len(target)],
                expected,
                rtol=1e-5,
                atol=1e-5,
            )
            expected_alignment = similarity_matrix.calculate_alignment_data(
                (0, index + 1, expected)
            )
            actual_alignment = similarity_matrix.calculate_alignment_data(
                (0, index + 1, actual[index, :, :len(target)])
            )
            self.assertEqual(actual_alignment[3], expected_alignment[3])
            self.assertEqual(actual_alignment[5], expected_alignment[5])

    def test_embedding_store_uses_full_cache_only_when_budget_allows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, "embeddings.h5")
            with h5py.File(h5_path, "w") as hf:
                group = hf.create_group("embeddings")
                group.create_dataset("a", data=np.ones((2, 4), dtype=np.float16))
                group.create_dataset("b", data=np.ones((3, 4), dtype=np.float16))

            with mock.patch.object(
                alignment_engine,
                "system_memory_bytes",
                return_value=(64 * alignment_engine.GIB, 48 * alignment_engine.GIB),
            ):
                cached = alignment_engine.EmbeddingTileStore(
                    h5_path, ["a", "b"], 1
                )
                tiled = alignment_engine.EmbeddingTileStore(
                    h5_path, ["a", "b"], 0
                )

        self.assertTrue(cached.fully_cached)
        self.assertFalse(tiled.fully_cached)
        np.testing.assert_array_equal(cached.get(1), np.ones((3, 4), np.float16))

    def test_host_cache_auto_and_numeric_caps_preserve_system_reserve(self):
        with mock.patch.object(
            alignment_engine,
            "system_memory_bytes",
            return_value=(64 * alignment_engine.GIB, 48 * alignment_engine.GIB),
        ):
            self.assertEqual(
                alignment_engine.resolve_host_cache_bytes("auto"),
                32 * alignment_engine.GIB,
            )
            self.assertEqual(
                alignment_engine.resolve_host_cache_bytes(40),
                32 * alignment_engine.GIB,
            )
            self.assertEqual(alignment_engine.resolve_host_cache_bytes(0), 0)

        with mock.patch.object(
            alignment_engine,
            "system_memory_bytes",
            return_value=(512 * alignment_engine.GIB, 400 * alignment_engine.GIB),
        ):
            self.assertEqual(
                alignment_engine.resolve_host_cache_bytes("auto"),
                128 * alignment_engine.GIB,
            )
            self.assertEqual(
                alignment_engine.resolve_host_cache_bytes(200),
                128 * alignment_engine.GIB,
            )
            self.assertEqual(
                alignment_engine.resolve_host_cache_bytes(96),
                96 * alignment_engine.GIB,
            )

    def test_cuda_memory_plan_divides_matrix_pool_across_lane_slots(self):
        memory_info = (16 << 30, 16 << 30)
        one_lane = alignment_engine.cuda_memory_plan(
            torch.device("cuda:0"),
            lanes=1,
            memory_info=memory_info,
        )
        four_lanes = alignment_engine.cuda_memory_plan(
            torch.device("cuda:0"),
            lanes=4,
            memory_info=memory_info,
        )
        self.assertEqual(one_lane.inflight_slots, 2)
        self.assertEqual(four_lanes.inflight_slots, 8)
        self.assertEqual(one_lane.matrix_pool_bytes, four_lanes.matrix_pool_bytes)
        self.assertLessEqual(
            abs(one_lane.matrix_bytes - four_lanes.matrix_bytes * 4),
            4,
        )

    def test_cuda_memory_plan_accepts_safe_benchmark_profiles(self):
        memory_info = (16 << 30, 16 << 30)
        plans = [
            alignment_engine.cuda_memory_plan(
                torch.device("cuda:0"),
                lanes=2,
                memory_info=memory_info,
                tile_fraction=tile_fraction,
                matrix_fraction=matrix_fraction,
            )
            for tile_fraction, matrix_fraction in (
                (0.20, 0.60),
                (0.30, 0.50),
                (0.40, 0.40),
            )
        ]
        self.assertLess(plans[0].tile_cache_bytes, plans[2].tile_cache_bytes)
        self.assertGreater(plans[0].matrix_bytes, plans[2].matrix_bytes)
        with self.assertRaisesRegex(ValueError, "at most 80%"):
            alignment_engine.cuda_memory_plan(
                torch.device("cuda:0"),
                lanes=2,
                memory_info=memory_info,
                tile_fraction=0.50,
                matrix_fraction=0.40,
            )

    def test_mps_memory_snapshot_caps_working_set_by_available_system_ram(self):
        backend = alignment_engine.AcceleratorBackend(torch.device("mps"))
        with mock.patch.object(
            alignment_engine.torch.mps,
            "recommended_max_memory",
            return_value=12 * alignment_engine.GIB,
            create=True,
        ), mock.patch.object(
            alignment_engine.torch.mps,
            "driver_allocated_memory",
            return_value=2 * alignment_engine.GIB,
            create=True,
        ), mock.patch.object(
            alignment_engine,
            "system_memory_bytes",
            return_value=(16 * alignment_engine.GIB, 8 * alignment_engine.GIB),
        ):
            snapshot = backend.memory_snapshot()

        self.assertEqual(snapshot.backend, "mps")
        self.assertTrue(snapshot.unified_memory)
        self.assertEqual(snapshot.free_bytes, 8 * alignment_engine.GIB)
        self.assertEqual(snapshot.reserve_bytes, int(12 * alignment_engine.GIB * 0.20))

    def test_mps_tiling_is_disabled_for_missing_or_invalid_memory_apis(self):
        backend = alignment_engine.AcceleratorBackend(torch.device("mps"))
        with mock.patch.object(
            alignment_engine.torch.mps,
            "recommended_max_memory",
            None,
            create=True,
        ):
            supported, reason = backend.supports_tiled(require_memory=True)
        self.assertFalse(supported)
        self.assertIn("recommended_max_memory", reason)

        with mock.patch.object(
            alignment_engine.torch.mps,
            "recommended_max_memory",
            return_value=0,
            create=True,
        ), mock.patch.object(
            alignment_engine.torch.mps,
            "driver_allocated_memory",
            return_value=0,
            create=True,
        ):
            supported, reason = backend.supports_tiled(require_memory=True)
        self.assertFalse(supported)
        self.assertIn("invalid memory", reason)

        with mock.patch.object(
            alignment_engine.torch.mps,
            "recommended_max_memory",
            return_value=4 * alignment_engine.GIB,
            create=True,
        ), mock.patch.object(
            alignment_engine.torch.mps,
            "driver_allocated_memory",
            return_value=3 * alignment_engine.GIB,
            create=True,
        ), mock.patch.object(
            alignment_engine,
            "system_memory_bytes",
            return_value=(8 * alignment_engine.GIB, alignment_engine.GIB),
        ):
            supported, reason = backend.supports_tiled(require_memory=True)
        self.assertFalse(supported)
        self.assertIn("no safely usable", reason)

    def test_mps_memory_plan_uses_one_device_resident_matrix_slot(self):
        plan = alignment_engine.accelerator_memory_plan(
            torch.device("mps"),
            lanes=1,
            memory_info=(8 * alignment_engine.GIB, 12 * alignment_engine.GIB),
        )
        self.assertEqual(plan.inflight_slots, 1)
        self.assertEqual(plan.matrix_bytes, plan.matrix_pool_bytes)

    def test_adaptive_tile_candidates_are_dtype_aware_and_deduplicated(self):
        store = mock.Mock(
            feature_dimension=2,
            float32_bytes=[900, 900, 900, 900],
        )
        store.block_ids = None
        tasks = [(0, 2, "a", "c"), (1, 3, "b", "d")]
        snapshot = alignment_engine.AcceleratorMemorySnapshot(
            backend="cuda",
            free_bytes=10000,
            total_bytes=10000,
            reserve_bytes=0,
            source="test",
        )

        fp32 = alignment_engine.build_adaptive_tile_plans(
            tasks,
            store=store,
            lengths=[2, 2, 2, 2],
            device=torch.device("cuda:0"),
            lane_candidates=[1],
            memory_snapshot=snapshot,
            compute_element_bytes=4,
        )
        bf16_ready = alignment_engine.build_adaptive_tile_plans(
            tasks,
            store=store,
            lengths=[2, 2, 2, 2],
            device=torch.device("cuda:0"),
            lane_candidates=[1],
            memory_snapshot=snapshot,
            compute_element_bytes=2,
        )

        self.assertTrue(fp32)
        self.assertTrue(bf16_ready)
        self.assertTrue(all(plan.memory_plan.compute_element_bytes == 2 for plan in bf16_ready))
        self.assertLessEqual(
            min(plan.estimate.embedding_reload_bytes for plan in bf16_ready),
            min(plan.estimate.embedding_reload_bytes for plan in fp32),
        )
        self.assertEqual(
            len({(plan.lanes, plan.estimate.schedule_signature) for plan in fp32}),
            len(fp32),
        )

    def test_adaptive_tile_candidates_reject_an_oversized_embedding(self):
        store = mock.Mock(feature_dimension=2, float32_bytes=[9000, 100])
        store.block_ids = None
        snapshot = alignment_engine.AcceleratorMemorySnapshot(
            backend="cuda",
            free_bytes=10000,
            total_bytes=10000,
            reserve_bytes=0,
            source="test",
        )
        plans = alignment_engine.build_adaptive_tile_plans(
            [(0, 1, "a", "b")],
            store=store,
            lengths=[2, 2],
            device=torch.device("cuda:0"),
            lane_candidates=[1, 2],
            memory_snapshot=snapshot,
        )
        self.assertEqual(plans, [])

    def test_vram_estimator_uses_explicit_benchmark_plan(self):
        store = mock.Mock(
            feature_dimension=8,
            float32_bytes=[64, 64],
        )
        store.block_ids.return_value = np.zeros(2, dtype=np.int32)
        plan = alignment_engine.cuda_memory_plan(
            torch.device("cuda:0"),
            lanes=2,
            memory_info=(16 << 30, 16 << 30),
            tile_fraction=0.20,
            matrix_fraction=0.60,
        )
        estimate = alignment_engine.estimate_cuda_working_set(
            [(0, 1, "a", "b")],
            store=store,
            lengths=[2, 2],
            device=torch.device("cuda:0"),
            lanes=2,
            memory_plan_override=plan,
        )
        self.assertEqual(estimate.per_microbatch_bytes, plan.matrix_bytes)

    def test_vram_estimator_rejects_unsafe_lane_count_before_cuda(self):
        store = mock.Mock(
            feature_dimension=128,
            float32_bytes=[4000 * 128 * 4] * 2,
        )
        store.block_ids.return_value = np.zeros(2, dtype=np.int32)
        tasks = [(0, 1, "a", "b")]
        memory_info = (16 << 30, 16 << 30)
        one_lane = alignment_engine.estimate_cuda_working_set(
            tasks,
            store=store,
            lengths=[4000, 4000],
            device=torch.device("cuda:0"),
            lanes=1,
            variant="tiled",
            memory_info=memory_info,
        )
        eight_lanes = alignment_engine.estimate_cuda_working_set(
            tasks,
            store=store,
            lengths=[4000, 4000],
            device=torch.device("cuda:0"),
            lanes=8,
            variant="tiled",
            memory_info=memory_info,
        )
        self.assertTrue(one_lane.feasible)
        self.assertFalse(eight_lanes.feasible)
        self.assertIn("per-slot budget", eight_lanes.reason)

    def test_microbatch_budget_includes_padded_embedding_tensor(self):
        tasks = [(0, 1, "a", "b"), (0, 2, "a", "c")]
        batches = list(
            alignment_engine._length_microbatches(
                tasks,
                lengths=[10, 100, 100],
                row_length=10,
                matrix_budget=6 * 1024 * 1024,
                feature_dimension=10000,
            )
        )
        self.assertEqual([len(batch) for batch in batches], [1, 1])

    def test_precision_comparison_requires_lengths_and_per_residue_scores(self):
        baseline = [(0, 1, 10.0, 10, 20.0, 10)]
        close = [(0, 1, 10.005, 10, 20.005, 10)]
        changed_length = [(0, 1, 10.005, 9, 20.005, 10)]
        far = [(0, 1, 10.02, 10, 20.0, 10)]

        self.assertTrue(
            alignment_engine.compare_precision_results(baseline, close)[0]
        )
        self.assertFalse(
            alignment_engine.compare_precision_results(
                baseline, changed_length
            )[0]
        )
        self.assertFalse(
            alignment_engine.compare_precision_results(baseline, far)[0]
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_tiled_cuda_pipeline_matches_scalar_pair_set(self):
        rng = np.random.default_rng(20260822)
        headers = [f"sequence_{index}" for index in range(6)]
        lengths = [5, 6, 7, 8, 9, 10]
        arrays = {
            header: rng.normal(size=(length, 32)).astype(np.float32)
            for header, length in zip(headers, lengths)
        }
        tasks = [
            (row, column, headers[row], headers[column])
            for row in range(len(headers))
            for column in range(row + 1, len(headers))
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, "embeddings.h5")
            with h5py.File(h5_path, "w") as hf:
                group = hf.create_group("embeddings")
                for header in headers:
                    group.create_dataset(header, data=arrays[header])
            store = alignment_engine.EmbeddingTileStore(h5_path, headers, 0)
            scalar_cuda = similarity_matrix._run_accelerated_pipeline(
                tasks,
                workers=2,
                input_h5=h5_path,
                device=torch.device("cuda:0"),
                batch_id=0,
                accelerator_workers=2,
                show_progress=False,
                matmul_precision="float32",
            )
            scalar_cuda_tf32 = similarity_matrix._run_accelerated_pipeline(
                tasks,
                workers=2,
                input_h5=h5_path,
                device=torch.device("cuda:0"),
                batch_id=0,
                accelerator_workers=2,
                show_progress=False,
                matmul_precision="tf32",
            )
            adaptive_plans = alignment_engine.build_adaptive_tile_plans(
                tasks,
                store=store,
                lengths=lengths,
                device=torch.device("cuda:0"),
                lane_candidates=[2],
            )
            self.assertTrue(adaptive_plans)
            selected_memory_plan = adaptive_plans[0].memory_plan
            tiled = alignment_engine.run_tiled_cuda_pipeline(
                tasks,
                store=store,
                lengths=lengths,
                device=torch.device("cuda:0"),
                workers=2,
                lanes=2,
                alignment_callback=similarity_matrix.calculate_alignment_data,
                precision="float32",
                memory_plan_override=selected_memory_plan,
            )
            tf32 = alignment_engine.run_tiled_cuda_pipeline(
                tasks,
                store=store,
                lengths=lengths,
                device=torch.device("cuda:0"),
                workers=2,
                lanes=2,
                alignment_callback=similarity_matrix.calculate_alignment_data,
                precision="tf32",
                memory_plan_override=selected_memory_plan,
            )

        scalar = []
        for row, column, header_i, header_j in tasks:
            matrix = similarity_matrix.compute_score_matrix_torch(
                arrays[header_i], arrays[header_j], torch.device("cpu")
            )
            scalar.append(
                similarity_matrix.calculate_alignment_data(
                    (row, column, matrix)
                )
            )

        tiled_by_pair = {result[:2]: result for result in tiled}
        scalar_by_pair = {result[:2]: result for result in scalar}
        self.assertEqual(tiled_by_pair.keys(), scalar_by_pair.keys())
        for pair, expected in scalar_by_pair.items():
            actual = tiled_by_pair[pair]
            self.assertEqual((actual[3], actual[5]), (expected[3], expected[5]))
            np.testing.assert_allclose(
                [actual[2], actual[4]],
                [expected[2], expected[4]],
                rtol=1e-5,
                atol=1e-3,
            )
        self.assertTrue(
            alignment_engine.compare_precision_results(tiled, tf32)[0]
        )
        self.assertTrue(
            alignment_engine.compare_precision_results(
                scalar_cuda,
                scalar_cuda_tf32,
            )[0]
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
            with h5py.File(input_h5, "w") as hf:
                embeddings = hf.create_group("embeddings")
                for header in ("a", "b", "c"):
                    embeddings.create_dataset(
                        header, data=np.ones((2, 2), dtype=np.float32)
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

    def test_row_local_sample_retains_multiple_pairs_per_selected_row(self):
        headers = [f"h{index}" for index in range(8)]
        lengths = [1, 2, 4, 8, 16, 32, 64, 128]
        computed = np.zeros((8, 8), dtype=bool)
        sample = similarity_matrix._representative_row_local_pending_pairs(
            headers, lengths, computed, None, 28, 16
        )
        counts = {}
        for row, _column, _header_i, _header_j in sample:
            counts[row] = counts.get(row, 0) + 1
        self.assertEqual(len(sample), 16)
        self.assertTrue(any(count > 1 for count in counts.values()))

    def test_confirmation_sample_matches_production_pair_order(self):
        headers = ["a", "b", "c", "d"]
        computed = np.zeros((4, 4), dtype=bool)
        computed[0, 2] = True
        required = np.triu(np.ones((4, 4), dtype=bool), k=1)
        required[1, 3] = False
        sample = similarity_matrix._first_pending_pairs(
            headers,
            computed,
            required,
            num_tasks=4,
            limit=3,
        )
        self.assertEqual(
            [(row, column) for row, column, _left, _right in sample],
            [(0, 1), (0, 3), (1, 2)],
        )

    def test_sparse_resume_counts_upper_triangle_pairs_once(self):
        required = np.triu(np.ones((4, 4), dtype=bool), k=1)
        computed = required | required.T

        self.assertEqual(
            similarity_matrix._count_computed_required_pairs(computed, required),
            6,
        )

        computed[1, 3] = False
        computed[3, 1] = False
        self.assertEqual(
            similarity_matrix._count_computed_required_pairs(computed, required),
            5,
        )

        self.assertEqual(
            similarity_matrix._count_computed_required_pairs(computed, None),
            5,
        )

    def test_shared_benchmark_sampler_builds_disjoint_global_halves(self):
        headers = [f"h{index}" for index in range(12)]
        lengths = [20 + index * 7 for index in range(12)]
        columns = {
            row: np.arange(row + 1, len(headers), dtype=np.int64)
            for row in range(len(headers))
        }
        counts = np.asarray([len(columns[row]) for row in range(len(headers))])

        first = alignment_engine.matched_benchmark_task_halves(
            headers,
            lengths,
            counts,
            lambda row: columns[row],
            half_pairs=20,
            row_limit=6,
        )
        second = alignment_engine.matched_benchmark_task_halves(
            headers,
            lengths,
            counts,
            lambda row: columns[row],
            half_pairs=20,
            row_limit=6,
        )

        self.assertEqual(first, second)
        warmup, timed = first
        self.assertEqual(len(warmup), 20)
        self.assertEqual(len(timed), 20)
        self.assertTrue(set(warmup).isdisjoint(timed))
        self.assertEqual(warmup, sorted(warmup, key=lambda task: task[:2]))
        self.assertEqual(timed, sorted(timed, key=lambda task: task[:2]))
        self.assertGreater(len({task[0] for task in warmup + timed}), 1)
        self.assertEqual(
            Counter(task[0] for task in warmup),
            Counter(task[0] for task in timed),
        )
        warmup_pairs = {(task[0], task[1]) for task in warmup}
        timed_pairs = {(task[0], task[1]) for task in timed}
        for row in sorted({task[0] for task in warmup + timed}):
            selected = sorted(
                (
                    task for task in warmup + timed
                    if task[0] == row
                ),
                key=lambda task: (lengths[row] * lengths[task[1]], task[1]),
            )
            for offset in range(0, len(selected), 2):
                adjacent = {
                    (selected[offset][0], selected[offset][1]),
                    (selected[offset + 1][0], selected[offset + 1][1]),
                }
                self.assertEqual(len(adjacent & warmup_pairs), 1)
                self.assertEqual(len(adjacent & timed_pairs), 1)

    def test_shared_benchmark_sampler_times_one_remaining_pair(self):
        warmup, timed = alignment_engine.matched_benchmark_task_halves(
            ["a", "b"],
            [10, 20],
            np.asarray([1, 0]),
            lambda row: np.asarray([1]) if row == 0 else np.asarray([]),
            half_pairs=4096,
        )
        self.assertEqual(warmup, [])
        self.assertEqual(timed, [(0, 1, "a", "b")])

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

        with mock.patch.object(
            similarity_matrix, "ACCELERATOR_CONFIRM_PAIRS", 3
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
            similarity_matrix, "ACCELERATOR_TUNE_PAIRS", 2
        ), mock.patch.object(
            similarity_matrix, "ACCELERATOR_CONFIRM_PAIRS", 6
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
        self.assertNotIn("[Tiles] Screening", output.getvalue())
        self.assertNotIn("Confirming Test CUDA", output.getvalue())
        for call in scalar_pipeline.call_args_list + tiled_pipeline.call_args_list:
            self.assertEqual(call.args[0], tasks)
            self.assertIn("benchmark_trial", call.kwargs)

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

    def test_tiled_cuda_oom_retries_then_uses_scalar_without_committed_duplicates(self):
        tasks = [(0, 1, "a", "b")]
        expected = [(0, 1, 1.0, 1, 2.0, 1)]
        sink = mock.Mock()
        sink.checkpoint.return_value = 0
        memory_plan = mock.Mock(matrix_bytes=256 * 1024 * 1024)

        with mock.patch.object(
            similarity_matrix,
            "run_tiled_accelerator_pipeline",
            side_effect=torch.cuda.OutOfMemoryError("simulated"),
        ) as tiled, mock.patch.object(
            similarity_matrix,
            "cuda_memory_plan",
            return_value=memory_plan,
        ), mock.patch.object(
            similarity_matrix,
            "_run_accelerated_pipeline",
            return_value=expected,
        ) as scalar, mock.patch.object(
            similarity_matrix,
            "tqdm",
            return_value=mock.Mock(),
        ), redirect_stdout(io.StringIO()):
            actual = similarity_matrix.process_accelerated_tasks(
                tasks,
                workers=2,
                input_h5="unused.h5",
                device=torch.device("cuda:0"),
                batch_id=1,
                accelerator_workers=1,
                embedding_store=mock.Mock(),
                sequence_lengths=[2, 2],
                execution_variant="tiled",
                result_callback=sink,
            )

        self.assertEqual(actual, expected)
        self.assertEqual(tiled.call_count, 4)
        self.assertEqual(sink.rollback.call_count, 4)
        scalar.assert_called_once()
        self.assertIs(scalar.call_args.kwargs["result_callback"], sink)

    def test_selected_adaptive_plan_oom_is_returned_to_ranked_fallback(self):
        tasks = [(0, 1, "a", "b")]
        sink = mock.Mock()
        sink.checkpoint.return_value = 0
        memory_plan = mock.Mock(matrix_bytes=256 * 1024 * 1024)

        with mock.patch.object(
            similarity_matrix,
            "run_tiled_accelerator_pipeline",
            side_effect=torch.cuda.OutOfMemoryError("simulated"),
        ) as tiled, mock.patch.object(
            similarity_matrix,
            "_run_accelerated_pipeline",
        ) as scalar, mock.patch.object(
            similarity_matrix,
            "tqdm",
            return_value=mock.Mock(),
        ), self.assertRaisesRegex(RuntimeError, "plan remained out of memory"):
            similarity_matrix.process_accelerated_tasks(
                tasks,
                workers=2,
                input_h5="unused.h5",
                device=torch.device("cuda:0"),
                batch_id=1,
                accelerator_workers=1,
                embedding_store=mock.Mock(),
                sequence_lengths=[2, 2],
                execution_variant="tiled",
                result_callback=sink,
                memory_plan_override=memory_plan,
            )

        self.assertEqual(tiled.call_count, 4)
        self.assertEqual(sink.rollback.call_count, 4)
        scalar.assert_not_called()

    def test_ranked_batch_runner_tries_the_next_confirmed_plan(self):
        cuda = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cuda:0", "CUDA", torch.device("cuda:0"), "cuda"
        )
        cpu = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        ranked = [
            similarity_matrix.Hardware_Utils.BenchmarkResult(
                cuda, 100.0, lanes=2, variant="tiled"
            ),
            similarity_matrix.Hardware_Utils.BenchmarkResult(
                cpu, 10.0, lanes=1, variant="scalar"
            ),
        ]
        with mock.patch.object(
            similarity_matrix, "DEVICE_SELECTION", "auto"
        ), mock.patch.object(
            similarity_matrix,
            "process_batch",
            side_effect=[RuntimeError("simulated OOM"), None],
        ) as process, redirect_stdout(io.StringIO()):
            active = similarity_matrix._run_batch_with_ranked_plans(
                ranked, 0, [(0, 1, "a", "b")], 0, 1, "unused.h5", "checksum"
            )

        self.assertEqual(active, 1)
        self.assertEqual(process.call_count, 2)
        self.assertEqual(
            process.call_args_list[0].kwargs["device"], torch.device("cuda:0")
        )
        self.assertEqual(
            process.call_args_list[1].kwargs["device"], torch.device("cpu")
        )

    def test_tiled_xpu_uses_backend_neutral_runner_and_batch_callback(self):
        tasks = [(0, 1, "a", "b")]
        expected = [(0, 1, 1.0, 1, 2.0, 1)]
        sink = mock.Mock()
        backend = mock.Mock(device_type="xpu")

        with mock.patch.object(
            similarity_matrix,
            "tiled_accelerator_support",
            return_value=(True, "mock XPU support"),
        ), mock.patch.object(
            similarity_matrix,
            "get_accelerator_backend",
            return_value=backend,
        ), mock.patch.object(
            similarity_matrix,
            "run_tiled_accelerator_pipeline",
            return_value=expected,
        ) as tiled, mock.patch.object(
            similarity_matrix,
            "_run_accelerated_pipeline",
        ) as scalar, mock.patch.object(
            similarity_matrix,
            "tqdm",
            return_value=mock.Mock(),
        ):
            actual = similarity_matrix.process_accelerated_tasks(
                tasks,
                workers=2,
                input_h5="unused.h5",
                device=torch.device("xpu:0"),
                batch_id=7,
                accelerator_workers=2,
                embedding_store=mock.Mock(),
                sequence_lengths=[2, 2],
                execution_variant="tiled",
                result_callback=sink,
            )

        self.assertEqual(actual, expected)
        scalar.assert_not_called()
        tiled.assert_called_once()
        self.assertEqual(tiled.call_args.kwargs["device"], torch.device("xpu:0"))
        self.assertEqual(tiled.call_args.kwargs["lanes"], 2)
        self.assertIs(tiled.call_args.kwargs["result_callback"], sink)

    def test_tiled_xpu_oom_retries_then_falls_back_to_scalar_xpu(self):
        tasks = [(0, 1, "a", "b")]
        expected = [(0, 1, 1.0, 1, 2.0, 1)]
        backend = mock.Mock(device_type="xpu")
        backend.is_out_of_memory.return_value = True
        memory_plan = mock.Mock(matrix_bytes=256 * 1024 * 1024)

        with mock.patch.object(
            similarity_matrix,
            "tiled_accelerator_support",
            return_value=(True, "mock XPU support"),
        ), mock.patch.object(
            similarity_matrix,
            "get_accelerator_backend",
            return_value=backend,
        ), mock.patch.object(
            similarity_matrix,
            "run_tiled_accelerator_pipeline",
            side_effect=RuntimeError("XPU out of memory"),
        ) as tiled, mock.patch.object(
            similarity_matrix,
            "cuda_memory_plan",
            return_value=memory_plan,
        ), mock.patch.object(
            similarity_matrix,
            "_run_accelerated_pipeline",
            return_value=expected,
        ) as scalar, mock.patch.object(
            similarity_matrix,
            "tqdm",
            return_value=mock.Mock(),
        ), redirect_stdout(io.StringIO()):
            actual = similarity_matrix.process_accelerated_tasks(
                tasks,
                workers=2,
                input_h5="unused.h5",
                device=torch.device("xpu:0"),
                batch_id=8,
                accelerator_workers=1,
                embedding_store=mock.Mock(),
                sequence_lengths=[2, 2],
                execution_variant="tiled",
            )

        self.assertEqual(actual, expected)
        self.assertEqual(tiled.call_count, 4)
        self.assertEqual(backend.empty_cache.call_count, 4)
        scalar.assert_called_once()
        self.assertEqual(
            scalar.call_args.args[3], torch.device("xpu:0")
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_tiled_process_batch_streams_and_publishes_complete_cuda_results(self):
        rng = np.random.default_rng(20260823)
        headers = ["a", "b", "c", "d"]
        lengths = [5, 6, 7, 8]
        tasks = [
            (row, column, headers[row], headers[column])
            for row in range(4)
            for column in range(row + 1, 4)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            input_h5 = os.path.join(temp_dir, "embeddings.h5")
            results_dir = os.path.join(temp_dir, "batches")
            os.makedirs(results_dir)
            with h5py.File(input_h5, "w") as hf:
                group = hf.create_group("embeddings")
                for header, length in zip(headers, lengths):
                    group.create_dataset(
                        header,
                        data=rng.normal(size=(length, 16)).astype(np.float32),
                    )
            store = alignment_engine.EmbeddingTileStore(input_h5, headers, 0)
            with mock.patch.object(
                similarity_matrix, "RESULTS_DIR", results_dir
            ), mock.patch.object(
                store, "get", wraps=store.get
            ) as embedding_reads:
                similarity_matrix.process_batch(
                    tasks,
                    batch_id=0,
                    workers=2,
                    input_h5=input_h5,
                    embedding_checksum="checksum",
                    model_name="model",
                    gap_penalties=[-2.0, 0.0],
                    device=torch.device("cuda:0"),
                    accelerator_workers=2,
                    execution_variant="tiled",
                    matmul_precision="ieee_fp32",
                    embedding_store=store,
                    sequence_lengths=lengths,
                )
            self.assertEqual(embedding_reads.call_count, len(headers))
            output = os.path.join(results_dir, "batch_00000.h5")
            self.assertTrue(os.path.exists(output))
            self.assertFalse(os.path.exists(output + ".partial"))
            with h5py.File(output, "r") as hf:
                self.assertEqual(len(hf["i"]), len(tasks))
                self.assertEqual(hf.attrs["matmul_precision"], "ieee_fp32")


if __name__ == "__main__":
    unittest.main()
