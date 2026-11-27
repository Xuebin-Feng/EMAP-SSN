"""Deadline accounting and production-order contracts for alignment trials."""
import io
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
for module_dir in (SRC_DIR, SRC_DIR / "tools"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

# The tests package points tool imports at a missing settings file.
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as alignment
    import Embedding_Alignment_Engine as engine
    import Network_Injection as injection
    from utilities import Align_GPU_Kernels

from tests.alignment_fixtures import ImmediateExecutor  # noqa: E402


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
        original = computed.copy()
        expected = [(i, j, headers[i], headers[j]) for i in range(5)
                    for j in range(i + 1, 5) if not computed[i, j]]
        for limit in (0, 1, 3, 100):
            self.assertEqual(alignment._first_pending_pairs(
                headers, computed, len(expected), limit), expected[:limit])
        self.assertEqual(list(alignment._iter_pending_pairs(headers, computed)), expected)
        np.testing.assert_array_equal(computed, original)

        # Sparse prefiltered runs iterate sorted int64 required_keys with computed keys removed.
        required_keys = np.array(
            [i * 5 + j for i in range(5) for j in range(i + 1, 5) if (i, j) != (0, 2)],
            dtype=np.int64,
        )
        computed_keys = np.array([0 * 5 + 1], dtype=np.int64)
        already_done = alignment._locate_sorted_keys(computed_keys, required_keys)[1]
        pending_keys = required_keys[~already_done]
        sparse_expected = [pair for pair in expected if pair[:2] != (0, 2)]
        self.assertEqual(
            list(alignment._iter_pending_key_tasks(headers, pending_keys, 5)),
            sparse_expected,
        )

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

    def test_fused_gpu_trial_warmup_deadline_and_short_batch(self):
        """run_fused_gpu_alignment_pipeline warms up without counting, stops on deadline, and drains cleanly."""
        real_empty = torch.empty

        def cpu_safe_empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        class FakeStream:
            cuda_stream = 0

            def wait_event(self, _ev):
                return None

            def synchronize(self):
                return None

        for stop_on_deadline in (True, False):
            with self.subTest(stop_on_deadline=stop_on_deadline), tempfile.TemporaryDirectory() as folder:
                path = str(Path(folder) / "embeddings.h5")
                headers = list("abcdef")
                with h5py.File(path, "w") as hf:
                    for idx, h in enumerate(headers):
                        hf.create_dataset(
                            "embeddings/" + h,
                            data=np.full((4, 4), float(idx + 1), dtype=np.float32),
                        )
                store = engine.EmbeddingTileStore(path, headers, 0)
                tasks = [
                    (i, j, headers[i], headers[j])
                    for i in range(len(headers))
                    for j in range(i + 1, len(headers))
                ]
                now = [0.0]
                trial = engine.BenchmarkTrial(clock=lambda: now[0])
                warmup_launches = []
                timed_launches = []

                class FakeEvent:
                    def record(self, _stream=None):
                        return None

                    def synchronize(self):
                        if trial.started_at is not None:
                            if stop_on_deadline and now[0] >= 6.0:
                                now[0] = 8.0
                            elif not stop_on_deadline:
                                now[0] += 0.1

                class FakeKernels:
                    def launch(self, name, _grid, _block, args, _stream=None):
                        bucket = timed_launches if trial.started_at is not None else warmup_launches
                        bucket.append(name)
                        if name.startswith("align_dp_"):
                            num_p = int(args[21])
                            args[17][:num_p].fill_(1.25)
                            args[18][:num_p].fill_(4)
                            args[19][:num_p].fill_(0.5)
                            args[20][:num_p].fill_(3)
                            if trial.started_at is not None and stop_on_deadline:
                                now[0] = 6.0

                # tile_cache_bytes=256 -> per_block=128 B = 2 seqs/block -> 3 blocks, 6 tiles.
                plan = engine.AcceleratorMemoryPlan(
                    free_bytes=1 << 20, total_bytes=1 << 20, usable_bytes=1 << 20,
                    tile_cache_bytes=256, matrix_pool_bytes=1 << 16,
                    matrix_bytes=1 << 16, reserve_bytes=0, lanes=1, inflight_slots=1,
                )
                with mock.patch.object(Align_GPU_Kernels, "align_kernels", return_value=FakeKernels()), \
                        mock.patch.object(Align_GPU_Kernels.torch, "empty", side_effect=cpu_safe_empty), \
                        mock.patch.object(Align_GPU_Kernels.torch.cuda, "device", side_effect=lambda _d: nullcontext()), \
                        mock.patch.object(Align_GPU_Kernels.torch.cuda, "Stream", side_effect=lambda **_kw: FakeStream()), \
                        mock.patch.object(Align_GPU_Kernels.torch.cuda, "stream", side_effect=lambda _s: nullcontext()), \
                        mock.patch.object(Align_GPU_Kernels.torch.cuda, "current_stream", side_effect=lambda _d=None: FakeStream()), \
                        mock.patch.object(Align_GPU_Kernels.torch.cuda, "Event", FakeEvent), \
                        mock.patch.object(Align_GPU_Kernels.torch.cuda, "synchronize"), \
                        mock.patch.object(Align_GPU_Kernels.torch.cuda, "empty_cache"):
                    results = Align_GPU_Kernels.run_fused_gpu_alignment_pipeline(
                        tasks,
                        store=store,
                        lengths=[4] * len(headers),
                        device=torch.device("cpu"),
                        lanes=1,
                        global_gap=0.0,
                        local_gap=-2.0,
                        precision="float32",
                        memory_plan_override=plan,
                        benchmark_trial=trial,
                        cuda_matmul_precision_ctx=lambda _p: nullcontext(),
                        precision_compute_dtype_fn=engine.precision_compute_dtype,
                        precision_element_bytes_fn=engine.precision_element_bytes,
                        accelerator_memory_plan_fn=engine.accelerator_memory_plan,
                        partition_tiles_fn=engine._partition_tiles,
                    )

                self.assertGreater(len(warmup_launches), 0)
                self.assertGreater(len(timed_launches), 0)
                self.assertEqual(trial.completed, trial.submitted)
                self.assertEqual(len(results), trial.completed)
                if stop_on_deadline:
                    self.assertEqual(trial.stop_reason, "deadline")
                    self.assertEqual(trial.tiles, 1)
                    self.assertLess(trial.completed, len(tasks))
                    self.assertEqual(trial.elapsed, 8.0)
                    self.assertAlmostEqual(trial.rate, trial.completed / 8.0)
                else:
                    self.assertEqual(trial.stop_reason, "batch finished")
                    self.assertEqual(trial.completed, len(tasks))
                    self.assertGreater(trial.tiles, 1)
                    self.assertEqual(sorted(r[:2] for r in results), sorted(t[:2] for t in tasks))


if __name__ == "__main__":
    unittest.main()
