"""Deadline accounting and production-order contracts for alignment trials."""
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import torch

from tests.test_align_similarity_matrix_pipeline import (
    similarity_matrix as alignment, alignment_engine as engine, ImmediateExecutor,
)
from tests.test_network_injection_pipeline import network_injection as injection


class FirstBatchBenchmarkTests(unittest.TestCase):
    def test_trial_deadline_minimum_and_drain_accounting(self):
        now = [0.0]
        trial = engine.BenchmarkTrial(clock=lambda: now[0])
        trial.start()
        now[0] = 10.0
        self.assertTrue(trial.can_submit())
        trial.submitted = 3
        self.assertFalse(trial.can_submit())
        with self.assertRaisesRegex(RuntimeError, "drain"):
            trial.stop(20)
        trial.completed = 3
        now[0] = 12.0
        trial.stop(20)
        self.assertEqual(trial.rate, 0.25)
        self.assertEqual(trial.stop_reason, "deadline")
        self.assertEqual(trial.deadline, 5.0)

    def test_pending_batch_order_resume_filter_and_masks_unchanged(self):
        headers = list("abcde")
        computed = np.zeros((5, 5), dtype=bool)
        computed[0, 1] = computed[1, 0] = True
        required = np.triu(np.ones((5, 5), dtype=bool), 1)
        required[0, 2] = False
        original = computed.copy(), required.copy()
        expected = [(i, j, headers[i], headers[j]) for i in range(5)
                    for j in range(i + 1, 5) if required[i, j] and not computed[i, j]]
        for limit in (0, 1, 3, 100):
            self.assertEqual(alignment._first_pending_pairs(
                headers, computed, required, len(expected), limit), expected[:limit])
        self.assertEqual(list(alignment._iter_pending_pairs(headers, computed, required)), expected)
        np.testing.assert_array_equal(computed, original[0])
        np.testing.assert_array_equal(required, original[1])

    def test_cpu_trial_bounds_queue_and_drains_without_warmup_counts(self):
        now = [0.0]
        trial = engine.BenchmarkTrial(clock=lambda: now[0])
        outstanding = [0]
        peaks = []
        chunks = []

        class Pool:
            def map_async(self, callback, chunk, chunksize):
                chunks.append(list(chunk))
                outstanding[0] += 1
                peaks.append(outstanding[0])
                def get():
                    outstanding[0] -= 1
                    if trial.started_at is not None:
                        now[0] += 6.0
                    return [callback(t) for t in chunk]
                return SimpleNamespace(get=get)

        results = engine.run_bounded_cpu_trial(Pool(), lambda t: t, list(range(100)), 2, trial)
        self.assertEqual(results, list(range(20)))
        self.assertEqual(trial.submitted, 20)
        self.assertEqual(trial.completed, 20)
        self.assertEqual(trial.elapsed, 12.0)
        self.assertEqual(max(peaks), 2)
        self.assertEqual(outstanding[0], 0)
        self.assertTrue(all(len(chunk) <= 10 for chunk in chunks))
        self.assertEqual(chunks[:4], [list(range(10)), list(range(10, 20)), list(range(20, 30)), [30, 31]])

    def test_injection_pending_order_and_resume_state_unchanged(self):
        from itertools import islice, chain
        headers = list("abcde")
        required = {i * 5 + j for i in range(5) for j in range(i + 1, 5)}
        required.remove(2)
        old = {1, 3}
        computed = {8}
        expected = [(i, j, headers[i], headers[j]) for i in range(5)
                    for j in range(i + 1, 5)
                    if i * 5 + j in required - old - computed]
        original = required.copy(), old.copy(), computed.copy()
        for batch_size in (1, 3, 100):
            pending = injection._iter_pending_pairs(headers, required, old, computed)
            first = list(islice(pending, batch_size))
            self.assertEqual(first, expected[:batch_size])
            self.assertEqual(list(chain(first, pending)), expected)
        self.assertEqual((required, old, computed), original)

    def test_cpu_short_batch_and_failure(self):
        class Pool:
            def map_async(self, callback, chunk, chunksize):
                return SimpleNamespace(get=lambda: [callback(t) for t in chunk])
        trial = engine.BenchmarkTrial()
        self.assertEqual(engine.run_bounded_cpu_trial(Pool(), lambda t: t, [1], 2, trial), [1])
        self.assertEqual(trial.completed, 1)
        self.assertEqual(trial.stop_reason, "batch finished")
        failed = engine.BenchmarkTrial()
        def callback(t):
            if failed.started_at is not None:
                raise ValueError("bad pair")
            return t
        with self.assertRaisesRegex(ValueError, "bad pair"):
            engine.run_bounded_cpu_trial(Pool(), callback, [1], 1, failed)
        self.assertIsNone(failed.stopped_at)

    def test_scalar_trials_restart_prefix_and_count_cpu_completion(self):
        for tool in (alignment, injection):
            with self.subTest(tool=tool.__name__), tempfile.TemporaryDirectory() as folder:
                path = str(Path(folder) / "embeddings.h5")
                with h5py.File(path, "w") as hf:
                    for h in "abcde":
                        hf.create_dataset("embeddings/" + h, data=np.ones((2, 2), np.float32))
                tasks = [(0, j, "a", h) for j, h in enumerate("bcde", 1)]
                now = [0.0]
                trial = engine.BenchmarkTrial(clock=lambda: now[0])
                calls = []
                def score(args):
                    calls.append((trial.started_at is not None, args[:2]))
                    if trial.started_at is not None:
                        now[0] = 6.0
                    return args[0], args[1], np.zeros((2, 2), np.float32)
                def finish(args):
                    if trial.started_at is not None:
                        now[0] = 8.0
                    return args[0], args[1], 0, 0, 0, 0
                with mock.patch.object(tool, "_compute_accelerated_matrix", side_effect=score), \
                     mock.patch.object(tool, "calculate_alignment_data", side_effect=finish):
                    results = tool._run_accelerated_pipeline(
                        tasks, 1, path, torch.device("cpu"), -1, 1, False,
                        benchmark_trial=trial,
                    )
                self.assertEqual([r[:2] for r in results], [(0, 1)])
                self.assertEqual([pair for timed, pair in calls if not timed], [t[:2] for t in tasks])
                self.assertEqual(trial.completed, 1)
                self.assertEqual(trial.elapsed, 8.0)

    def test_tiled_slow_preparation_finishes_one_microbatch_on_both_backends(self):
        for asynchronous in (True, False):
            with self.subTest(asynchronous=asynchronous), tempfile.TemporaryDirectory() as folder:
                path = str(Path(folder) / "embeddings.h5")
                headers = list("abcde")
                with h5py.File(path, "w") as hf:
                    for h in headers:
                        hf.create_dataset("embeddings/" + h, data=np.ones((2, 2), np.float32))
                store = engine.EmbeddingTileStore(path, headers, 0)
                tasks = [(0, j, "a", headers[j]) for j in range(1, 5)]
                now = [0.0]
                trial = engine.BenchmarkTrial(clock=lambda: now[0])
                event = SimpleNamespace(record=lambda s: None, query=lambda: True, synchronize=lambda: None)
                backend = SimpleNamespace(
                    supports_async_streams=asynchronous, supports_tiled=lambda **k: (True, "mock"),
                    create_stream=lambda: object(), stream_context=lambda s: nullcontext(),
                    create_event=lambda: event, empty_cache=lambda: None, synchronize=lambda: None,
                )
                plan = engine.CudaMemoryPlan(
                    free_bytes=1 << 30, total_bytes=1 << 30, usable_bytes=1 << 30,
                    tile_cache_bytes=1 << 20, matrix_pool_bytes=1 << 20,
                    matrix_bytes=1 << 19, reserve_bytes=0, lanes=1, inflight_slots=2,
                )
                load = store.load_indices
                def slow_load(*args):
                    if trial.started_at is not None:
                        now[0] = 10.0
                    return load(*args)
                def finish(args):
                    if trial.started_at is not None:
                        now[0] = 12.0
                    return args[0], args[1], 0, 0, 0, 0
                with mock.patch.object(engine, "get_accelerator_backend", return_value=backend), \
                     mock.patch.object(store, "load_indices", side_effect=slow_load), \
                     mock.patch.object(engine, "_to_normalized_accelerator", side_effect=lambda a, *rest: torch.as_tensor(a)), \
                     mock.patch.object(engine, "_length_microbatches", side_effect=lambda tasks, *rest: ([t] for t in tasks)), \
                     mock.patch.object(engine, "_batched_score_matrices", side_effect=lambda *args: torch.zeros((1, 2, 2))), \
                     mock.patch.object(engine, "ThreadPoolExecutor", ImmediateExecutor):
                    results = engine.run_tiled_accelerator_pipeline(
                        tasks, store=store, lengths=[2] * 5, device=torch.device("cpu"),
                        workers=1, lanes=1, alignment_callback=finish,
                        memory_plan_override=plan, benchmark_trial=trial,
                    )
                self.assertEqual([r[:2] for r in results], [(0, 1)])
                self.assertEqual((trial.completed, trial.tiles, trial.microbatches), (1, 1, 1))
                self.assertEqual(trial.elapsed, 12.0)


if __name__ == "__main__":
    unittest.main()
