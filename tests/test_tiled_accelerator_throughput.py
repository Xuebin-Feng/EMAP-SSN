import inspect
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.utilities import Embedding_Alignment_Engine as engine
from src.utilities.Alignment_Score_Kernels import (
    GlobalLocalScratch,
    global_local_scores,
)


class TiledThroughputUnitTests(unittest.TestCase):
    def test_pair_stream_preserves_external_ordinals_across_chunks(self):
        session = object.__new__(engine.TiledAcceleratorSession)
        session.store = SimpleNamespace(headers=["a", "b", "c"])
        calls = []

        def run(tasks, *, progress=None, result_callback=None, result_chunk_size=None):
            calls.append(len(tasks))
            results = [
                (
                    int(task[0]),
                    int(task[1]),
                    np.float32(1.0),
                    np.uint16(2),
                    np.float32(1.5),
                    np.uint16(2),
                )
                for task in tasks
            ]
            result_callback(results)
            return []

        session.run = run
        results = list(
            session.run_pair_stream(
                iter(((10, 0, 1), (20, 0, 2), (30, 1, 2))),
                chunk_size=2,
            )
        )
        self.assertEqual(calls, [2, 1])
        self.assertEqual([result[0] for result in results], [10, 20, 30])

    def test_compact_pair_tasks_preserve_historical_contract(self):
        headers = ["a", "b", "c"]
        tasks = engine.CompactPairTasks(3, headers)
        tasks.append(0, 2)
        tasks.append((1, 2, "ignored", "ignored"))
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0], (0, 2, "a", "c"))
        self.assertEqual(tasks[:], [(0, 2, "a", "c"), (1, 2, "b", "c")])

    def test_multirow_packet_keeps_ordinals_and_padding_bounded(self):
        lengths = [300, 306, 310, 312]
        tasks = [(0, 2), (1, 3), (0, 3), (1, 2)]
        packets = list(
            engine.iter_pair_work_packets(
                tasks,
                lengths,
                256 * engine.MIB,
                64,
            )
        )
        self.assertLess(len(packets), len(tasks))
        self.assertEqual(
            sorted(value for packet in packets for value in packet.ordinals),
            list(range(len(tasks))),
        )
        for packet in packets:
            self.assertLessEqual(
                packet.padded_cells,
                packet.real_cells * (1.0 + engine.PADDING_OVERHEAD_LIMIT),
            )

    def test_padded_multirow_scorer_matches_scalar_equations(self):
        rng = np.random.default_rng(20260823)
        arrays = [
            rng.normal(size=(length, 11)).astype(np.float32)
            for length in (5, 7, 6, 8)
        ]
        tensors = [
            torch.nn.functional.normalize(torch.from_numpy(array), dim=-1)
            for array in arrays
        ]
        actual = engine._batched_pair_score_matrices(
            [tensors[0], tensors[1]],
            [tensors[2], tensors[3]],
            [5, 7],
            [6, 8],
            query_bucket=7,
            target_bucket=8,
        ).numpy()
        for offset, (left, right) in enumerate(((0, 2), (1, 3))):
            expected = engine.compute_score_matrix_torch(
                arrays[left], arrays[right], torch.device("cpu")
            )
            np.testing.assert_allclose(
                actual[offset, :arrays[left].shape[0], :arrays[right].shape[0]],
                expected,
                rtol=2e-5,
                atol=2e-5,
            )

    def test_host_buffer_lease_reuses_only_after_last_chunk(self):
        pool = engine._HostBufferPool(2 * engine.MIB)
        tensor, pooled = pool.acquire((2, 16, 16), pin_memory=False)
        self.assertTrue(pooled)
        lease = engine._HostBufferLease(pool, tensor, pooled, 2)
        self.assertFalse(lease.release_reference())
        self.assertTrue(lease.release_reference())
        reused, reused_pooled = pool.acquire((2, 16, 16), pin_memory=False)
        self.assertTrue(reused_pooled)
        self.assertEqual(reused.data_ptr(), tensor.data_ptr())

    def test_thread_scratch_grows_once_and_preserves_kernel_results(self):
        rng = np.random.default_rng(17)
        matrix = rng.normal(size=(9, 13)).astype(np.float32)
        scratch = GlobalLocalScratch()
        first, grew_first = scratch.score(matrix, 0.0, -2.0)
        second, grew_second = scratch.score(matrix, 0.0, -2.0)
        expected = global_local_scores(matrix, 0.0, -2.0)
        self.assertTrue(grew_first)
        self.assertFalse(grew_second)
        np.testing.assert_allclose(first, expected)
        np.testing.assert_allclose(second, expected)

    def test_staged_tuner_narrows_lanes_before_other_axes(self):
        def memory(lanes):
            return engine.AcceleratorMemoryPlan(
                free_bytes=8 * engine.GIB,
                total_bytes=12 * engine.GIB,
                usable_bytes=6 * engine.GIB,
                tile_cache_bytes=2 * engine.GIB,
                matrix_pool_bytes=2 * engine.GIB,
                matrix_bytes=512 * engine.MIB,
                reserve_bytes=2 * engine.GIB,
                lanes=lanes,
                inflight_slots=lanes * 2,
            )

        calls = []

        def measure(plan, tasks):
            calls.append(plan)
            rate = 1000.0 - abs(plan.lanes - 2) * 100.0
            rate += plan.microbatch_workspace_bytes / engine.GIB
            rate += plan.cpu_chunk_size
            return {
                "rate": rate,
                "peak_memory_bytes": plan.microbatch_workspace_bytes,
                "results": (),
            }

        ranked, _observations = engine.benchmark_accelerator_execution_plans(
            lane_candidates=[1, 2, 4],
            memory_plan_factory=memory,
            workers=8,
            short_tasks=[(0, 1)] * 8,
            confirmation_tasks=[(0, 1)] * 16,
            remaining_pairs=1000,
            measure=measure,
            allow_compilation=False,
        )
        self.assertTrue(ranked)
        first_three = calls[:3]
        self.assertEqual([plan.lanes for plan in first_three], [1, 2, 4])
        self.assertTrue(
            all(
                plan.microbatch_workspace_bytes <= 512 * engine.MIB
                for plan in first_three
            )
        )

    def test_metrics_reports_padding_and_bottleneck(self):
        metrics = engine.AcceleratorPipelineMetrics(
            elapsed_seconds=10.0,
            cpu_queue_stall_seconds=2.0,
            padded_cells=120,
            real_cells=100,
        ).as_dict()
        self.assertEqual(metrics["bottleneck"], "cpu-dp-bound")
        self.assertAlmostEqual(metrics["padding_fraction"], 1.0 / 6.0)

    def test_shared_session_has_no_direct_backend_runtime_calls(self):
        source = inspect.getsource(engine.TiledAcceleratorSession)
        self.assertNotIn("torch.cuda", source)
        self.assertNotIn("torch.xpu", source)


if __name__ == "__main__":
    unittest.main()
