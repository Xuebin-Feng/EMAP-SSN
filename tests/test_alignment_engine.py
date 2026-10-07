"""Tests of Embedding_Alignment_Engine on its own: execution-mode and precision
settings, host and accelerator memory planning (CUDA, XPU, MPS), adaptive tile
plans, VRAM estimates, the shared benchmark sampler and the tiled pipeline's
phase handling. Tests that combine the engine with Align_Similarity_Matrix
callbacks live in test_align_similarity_matrix_pipeline.py.
"""
import os
import sys
import tempfile
import unittest
from collections import Counter
from concurrent.futures import Future
from unittest import mock

import h5py
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Embedding_Alignment_Engine as alignment_engine  # noqa: E402

from tests.alignment_fixtures import RecordingTrial  # noqa: E402


class EmbeddingAlignmentEngineTests(unittest.TestCase):
    """Engine functions exercised without any alignment tool module."""

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
        trial = RecordingTrial(events)
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
                    benchmark_trial=trial,
                )

        # The warm-up covers all four pairs (fewer than the warm-up minimum);
        # the timed phase then starts again from the first pair. Both phases
        # share one HDF5 handle and one stream; the cache is released once.
        trial_start = events.index("trial-start")
        trial_stop = events.index("trial-stop")
        aligned = [f"align-{column}" for column in range(1, 5)]
        self.assertEqual(len(results), 4)
        self.assertEqual(len(backend.streams), 1)
        self.assertEqual(opened.call_count, 1)
        self.assertCountEqual(events[:trial_start], aligned)
        self.assertCountEqual(events[trial_start + 1:trial_stop], aligned)
        self.assertEqual(events[trial_stop + 1:], ["empty-cache"])
        self.assertEqual((trial.submitted, trial.completed), (4, 4))

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


if __name__ == "__main__":
    unittest.main()
