"""SSEARCH's tiled CUDA path: the chunked, prefetching fixed-query pipeline.

The pipeline reads embeddings in chunks on a reader thread, copies each chunk up once, normalizes
every target as ``_to_normalized_cuda`` does into one packed buffer, gathers each microbatch's padded
targets in one operation, and aligns a whole microbatch per CPU task. Its results must equal those of
the per-target pipeline it replaced, bit for bit; the CUDA tests below check that against a sequential
reference built from the old pieces. The CPU tests cover the chunk plan, the reader and the store's
``read_into``.
"""

import io
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (
    os.path.join(PROJECT_ROOT, "src"),
    os.path.join(PROJECT_ROOT, "src", "tools"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Embedding_SSEARCH as ssearch
    import Embedding_Alignment_Engine as engine
from tools.tool_helpers.Tool_Pipeline import SearchTiming  # noqa: E402

RESIDUES = list("ACDEFGHIKLMNPQRSTVWY")
CUDA = torch.cuda.is_available()


class PlanUploadChunksTests(unittest.TestCase):
    def test_chunks_fill_a_buffer_at_aligned_offsets(self):
        chunks = engine.plan_upload_chunks([0, 1, 2, 3, 4], [10, 10, 10, 100, 5], chunk_bytes=32, alignment=16)
        self.assertEqual(chunks, [
            ([(0, 0, 10), (1, 16, 10)], 32),
            ([(2, 0, 10)], 16),
            ([(3, 0, 100)], 112),
            ([(4, 0, 5)], 16),
        ])

    def test_every_index_is_in_exactly_one_chunk_in_order(self):
        rng = np.random.default_rng(3)
        sizes = rng.integers(1, 5000, size=200).tolist()
        indices = sorted(rng.choice(200, size=120, replace=False).tolist())
        chunks = engine.plan_upload_chunks(indices, sizes, chunk_bytes=8192)
        flat = [entry for entries, _used in chunks for entry in entries]
        self.assertEqual([index for index, _offset, _size in flat], indices)
        for entries, used in chunks:
            self.assertLessEqual(used, 8192 if len(entries) > 1 else max(8192, used))
            for index, offset, size in entries:
                self.assertEqual(offset % engine.FIXED_QUERY_ALIGNMENT, 0)
                self.assertEqual(size, sizes[index])
                self.assertLessEqual(offset + size, used)
            offsets = [offset for _index, offset, _size in entries]
            self.assertEqual(offsets, sorted(offsets))

    def test_an_oversized_embedding_is_a_chunk_of_its_own(self):
        chunks = engine.plan_upload_chunks([5, 6, 7], {5: 20, 6: 1000, 7: 20}, chunk_bytes=64)
        self.assertEqual([[index for index, _o, _s in entries] for entries, _u in chunks], [[5], [6], [7]])

    def test_nothing_to_upload_is_no_chunks(self):
        self.assertEqual(engine.plan_upload_chunks([], []), [])


class FakeStore:
    """The part of EmbeddingTileStore the reader uses: dtypes, shapes, a path and read_into."""

    def __init__(self, shapes, dtype=np.float16, fail_at=None, delay=0.0):
        self.path = "fake.h5"
        self.shapes = list(shapes)
        self.dtypes = [np.dtype(dtype)] * len(shapes)
        self.source_bytes = [int(np.prod(s)) * np.dtype(dtype).itemsize for s in shapes]
        self.fail_at = fail_at
        self.delay = delay
        self.reads = []
        self.lock = threading.Lock()

    def read_into(self, index, out, group=None):
        if self.delay:
            time.sleep(self.delay)
        if index == self.fail_at:
            raise KeyError(f"embedding {index} is missing")
        out[...] = index + 1
        with self.lock:
            self.reads.append(index)
        return out


class FakeFile:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return {"embeddings": object()}

    def __exit__(self, *exc):
        return False


def reader_for(store, buffer_count, chunk_bytes=256):
    indices = list(range(len(store.shapes)))
    chunks = engine.plan_upload_chunks(indices, store.source_bytes, chunk_bytes)
    capacity = max(used for _entries, used in chunks)
    buffers = [np.zeros(capacity, dtype=np.uint8) for _ in range(buffer_count)]
    return engine._ChunkReader(store, chunks, buffers), chunks


class ChunkReaderTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(engine.h5py, "File", FakeFile)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_chunks_arrive_in_order_with_the_data_read_into_their_slot(self):
        store = FakeStore([(4, 8)] * 9)
        reader, chunks = reader_for(store, 3)
        reader.start()
        try:
            for number, (entries, used) in enumerate(chunks):
                got_number, got_entries, got_used, slot = reader.next_chunk()
                self.assertEqual((got_number, got_entries, got_used), (number, entries, used))
                for index, offset, size in entries:
                    values = reader.buffers[slot][offset:offset + size].view(np.float16)
                    self.assertTrue(np.all(values == index + 1))
                reader.release(slot)
        finally:
            reader.close()
        self.assertEqual(store.reads, list(range(9)))
        self.assertFalse(reader.is_alive())

    def test_reading_stops_while_every_buffer_holds_an_unreleased_chunk(self):
        store = FakeStore([(4, 8)] * 12)  # 64 bytes each: four per chunk of 256
        reader, chunks = reader_for(store, 2)
        self.assertEqual(len(chunks), 3)
        reader.start()
        try:
            time.sleep(0.3)
            self.assertEqual(len(store.reads), 8)  # two chunks read, the third waits for a slot
            first = reader.next_chunk()
            time.sleep(0.2)
            self.assertEqual(len(store.reads), 8)  # taking a chunk is not releasing its buffer
            reader.release(first[3])
            for _ in range(2):
                item = reader.next_chunk()
                reader.release(item[3])
            self.assertEqual(len(store.reads), 12)
        finally:
            reader.close()

    def test_a_failure_while_reading_surfaces_from_next_chunk(self):
        store = FakeStore([(4, 8)] * 9, fail_at=5)
        reader, _chunks = reader_for(store, 3)
        reader.start()
        try:
            first = reader.next_chunk()
            reader.release(first[3])
            with self.assertRaisesRegex(KeyError, "embedding 5 is missing"):
                reader.next_chunk()
        finally:
            reader.close()
        self.assertFalse(reader.is_alive())

    def test_close_stops_a_reader_waiting_for_a_free_buffer(self):
        store = FakeStore([(4, 8)] * 30)
        reader, _chunks = reader_for(store, 2)
        reader.start()
        time.sleep(0.2)
        started = time.perf_counter()
        reader.close()
        self.assertFalse(reader.is_alive())
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertLess(len(store.reads), 30)

    def test_a_close_before_the_reader_started_is_harmless(self):
        reader, _chunks = reader_for(FakeStore([(4, 8)] * 4), 2)
        reader.close()

    def test_the_consumer_can_work_while_it_waits(self):
        store = FakeStore([(4, 8)] * 4, delay=0.05)
        reader, _chunks = reader_for(store, 2)
        ticks = []
        reader.start()
        try:
            item = reader.next_chunk(idle=lambda: ticks.append(1))
            reader.release(item[3])
        finally:
            reader.close()
        self.assertGreater(len(ticks), 2)

    def test_a_reader_that_ends_early_is_an_error_not_a_hang(self):
        store = FakeStore([(4, 8)] * 4)
        reader, _chunks = reader_for(store, 2)
        reader.chunks = reader.chunks[:1]
        reader.start()
        try:
            item = reader.next_chunk()
            reader.release(item[3])
            with self.assertRaisesRegex(RuntimeError, "stopped before every chunk"):
                reader.next_chunk()
        finally:
            reader.close()


class ReadIntoTests(unittest.TestCase):
    def make_file(self, directory, dtype):
        path = os.path.join(directory, "embeddings.h5")
        rng = np.random.default_rng(1)
        with h5py.File(path, "w") as hf:
            group = hf.create_group("embeddings")
            for index, rows in enumerate((3, 7, 1, 12)):
                group.create_dataset(f"h{index}", data=rng.normal(size=(rows, 5)).astype(dtype))
        return path

    def test_it_reads_what_get_reads_for_every_stored_dtype(self):
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype.__name__), tempfile.TemporaryDirectory() as directory:
                path = self.make_file(directory, dtype)
                store = engine.EmbeddingTileStore(path, [f"h{i}" for i in range(4)], host_cache_setting=0)
                with h5py.File(path, "r", libver="latest", swmr=True) as hf:
                    for index in range(4):
                        out = np.empty(store.shapes[index], dtype=store.dtypes[index])
                        self.assertIs(store.read_into(index, out, hf["embeddings"]), out)
                        np.testing.assert_array_equal(out, store.get(index, hf["embeddings"]))
                        alone = np.empty_like(out)
                        store.read_into(index, alone)
                        np.testing.assert_array_equal(alone, out)

    def test_a_fully_cached_store_copies_from_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory, np.float16)
            store = engine.EmbeddingTileStore(path, [f"h{i}" for i in range(4)], host_cache_setting=1)
            self.assertTrue(store.fully_cached)
            with mock.patch.object(engine.h5py, "File", side_effect=AssertionError("opened the file")):
                for index in range(4):
                    out = np.empty(store.shapes[index], dtype=np.float16)
                    store.read_into(index, out)
                    np.testing.assert_array_equal(out, store.get(index))

    def test_a_missing_embedding_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory, np.float16)
            store = engine.EmbeddingTileStore(path, [f"h{i}" for i in range(4)], host_cache_setting=0)
            store.headers[2] = "absent"
            with h5py.File(path, "r", libver="latest", swmr=True) as hf, self.assertRaises(KeyError):
                store.read_into(2, np.empty(store.shapes[2], np.float16), hf["embeddings"])


class EstimateTests(unittest.TestCase):
    def test_the_vram_estimate_includes_the_staging_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "embeddings.h5")
            with h5py.File(path, "w") as hf:
                group = hf.create_group("embeddings")
                group.create_dataset("a", data=np.ones((3, 4), np.float16))
                group.create_dataset("b", data=np.ones((7, 4), np.float32))
            store = engine.EmbeddingTileStore(path, ["a", "b"], 0)

            def peak():
                return engine.estimate_fixed_query_cuda_working_set(
                    [(0, "a"), (1, "b")], query_embedding=np.ones((5, 4), np.float32), store=store,
                    lengths=[3, 7], device=torch.device("cuda"), lanes=2,
                    memory_info=(12 * 1024 ** 3, 16 * 1024 ** 3),
                ).projected_peak_bytes

            with mock.patch.object(engine, "FIXED_QUERY_CHUNK_BYTES", 1024):
                small = peak()
            with mock.patch.object(engine, "FIXED_QUERY_CHUNK_BYTES", 1024 + 4096):
                large = peak()
            self.assertEqual(large - small, 4096)
            # An embedding larger than the chunk sets the staging size.
            with mock.patch.object(engine, "FIXED_QUERY_CHUNK_BYTES", 1):
                self.assertEqual(peak() - small, 28 * 4 - 1024)


class FixedQueryHarness:
    """A small search on real CUDA, and a sequential reference built from the old per-target pieces."""

    FEATURES = 16
    QUERY_ROWS = 12

    def build(self, directory, targets=57, dtype=np.float16, mode="local", gap=-2.0, norm="longer_sequence", seed=0,
              length_range=(3, 41)):
        rng = np.random.default_rng(seed)
        path = os.path.join(directory, "embeddings.h5")
        lengths = rng.integers(*length_range, size=targets)
        query = rng.normal(size=(self.QUERY_ROWS, self.FEATURES)).astype(dtype)
        with h5py.File(path, "w") as hf:
            group = hf.create_group("embeddings")
            for index, length in enumerate(lengths):
                group.create_dataset(f"t{index}", data=rng.normal(size=(int(length), self.FEATURES)).astype(dtype))
        query_sequence = "".join(rng.choice(RESIDUES, size=self.QUERY_ROWS))
        tasks = [
            (index, f"t{index}", f"t{index}", query, mode, gap, norm, query_sequence,
             "".join(rng.choice(RESIDUES, size=int(length))))
            for index, length in enumerate(lengths)
        ]
        store = engine.EmbeddingTileStore(path, [task[2] for task in tasks], host_cache_setting=0)
        return path, tasks, store, [shape[0] for shape in store.shapes], query

    @staticmethod
    def plan(tile_bytes, matrix_bytes):
        return SimpleNamespace(tile_cache_bytes=tile_bytes, matrix_bytes=matrix_bytes)

    def reference(self, tasks, store, lengths, query, plan, precision="float32"):
        """The per-target pipeline, sequentially: padded copies per target, one finish per target."""
        device = torch.device("cuda:0")
        element_bytes = engine.precision_element_bytes(precision)
        block_ids = engine.fixed_query_block_ids(store, query, plan, element_bytes)
        grouped = {}
        for task in tasks:
            grouped.setdefault(int(block_ids[task[0]]), []).append(task)
        query_tensor = engine._to_normalized_accelerator(np.asarray(query), device, precision)
        results = []
        with torch.inference_mode(), engine.cuda_matmul_precision(precision),                 h5py.File(store.path, "r", libver="latest", swmr=True) as hf:
            for block_tasks in grouped.values():
                tensors = {
                    index: engine._to_normalized_accelerator(array, device, precision)
                    for index, array in store.load_indices({t[0] for t in block_tasks}, hf["embeddings"]).items()
                }
                pseudo = [(0, task[0], task) for task in block_tasks]
                for microbatch in engine._length_microbatches(
                    pseudo, lengths, self.QUERY_ROWS, plan.matrix_bytes, store.feature_dimension, element_bytes
                ):
                    target_lengths = [lengths[item[1]] for item in microbatch]
                    matrices = engine._batched_score_matrices(
                        query_tensor, [tensors[item[1]] for item in microbatch], target_lengths
                    ).cpu().numpy()
                    for offset, (item, length) in enumerate(zip(microbatch, target_lengths)):
                        task = item[2]
                        results.append(ssearch.finish_search((
                            task[0], task[1], self.QUERY_ROWS, length, task[4], task[5], task[6],
                            matrices[offset, :, :length], task[7], task[8],
                        )))
        return sorted(results, key=lambda row: row["index"])

    def run_pipeline(self, tasks, store, lengths, query, plan, lanes=2, workers=2, precision="float32", **extra):
        with mock.patch.object(engine, "cuda_memory_plan", return_value=plan):
            rows = engine.run_fixed_query_cuda_pipeline(
                tasks, query_embedding=query, store=store, lengths=lengths, device=torch.device("cuda:0"),
                workers=workers, lanes=lanes, batch_alignment_callback=ssearch.finish_search_batch,
                precision=precision, **extra,
            )
        return sorted(rows, key=lambda row: row["index"])


@unittest.skipUnless(CUDA, "needs a CUDA device")
class FixedQueryPipelineCudaTests(unittest.TestCase):
    def setUp(self):
        self.harness = FixedQueryHarness()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def assert_same_rows(self, got, want):
        self.assertEqual(len(got), len(want))
        for row_got, row_want in zip(got, want):
            self.assertEqual(row_got, row_want)
            self.assertEqual(
                {key: type(value) for key, value in row_got.items()},
                {key: type(value) for key, value in row_want.items()},
            )

    def test_results_equal_the_per_target_pipelines_for_every_lane_count_and_mode(self):
        plan = self.harness.plan(tile_bytes=14_000, matrix_bytes=40_000)  # several blocks, small microbatches
        for mode, gap, norm in (("local", -2.0, "longer_sequence"), ("global", 0.0, "alignment_length"),
                                ("global", -0.5, "shorter_sequence")):
            path, tasks, store, lengths, query = self.harness.build(self.directory.name, mode=mode, gap=gap, norm=norm)
            want = self.harness.reference(tasks, store, lengths, query, plan)
            blocks = len(set(engine.fixed_query_block_ids(store, query, plan, 4).tolist()))
            self.assertGreater(blocks, 3)
            for lanes in (1, 2, 3):
                with self.subTest(mode=mode, gap=gap, lanes=lanes):
                    self.assert_same_rows(self.harness.run_pipeline(tasks, store, lengths, query, plan, lanes=lanes), want)

    def test_float32_stored_embeddings_and_a_single_block(self):
        for dtype, tile in ((np.float32, 14_000), (np.float16, 10 ** 9)):
            with self.subTest(dtype=dtype.__name__, tile=tile):
                plan = self.harness.plan(tile_bytes=tile, matrix_bytes=60_000)
                path, tasks, store, lengths, query = self.harness.build(self.directory.name, dtype=dtype, seed=4)
                self.assert_same_rows(
                    self.harness.run_pipeline(tasks, store, lengths, query, plan),
                    self.harness.reference(tasks, store, lengths, query, plan),
                )

    def test_bf16_compute_is_the_same_as_before(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("this device has no bf16")
        plan = self.harness.plan(tile_bytes=9_000, matrix_bytes=40_000)
        path, tasks, store, lengths, query = self.harness.build(self.directory.name, seed=6)
        self.assert_same_rows(
            self.harness.run_pipeline(tasks, store, lengths, query, plan, precision="bf16"),
            self.harness.reference(tasks, store, lengths, query, plan, precision="bf16"),
        )

    def test_a_slow_lane_never_reads_a_block_that_was_overwritten(self):
        # Delay every microbatch on the device: if the next block's upload were not ordered after
        # the lanes' reads of the packed buffer, a slow lane would score the wrong embeddings.
        plan = self.harness.plan(tile_bytes=9_000, matrix_bytes=30_000)
        path, tasks, store, lengths, query = self.harness.build(self.directory.name, seed=8)
        want = self.harness.reference(tasks, store, lengths, query, plan)
        original = engine._batched_score_matrices

        def slow(*args, **kwargs):
            torch.cuda._sleep(20_000_000)
            return original(*args, **kwargs)

        with mock.patch.object(engine, "_batched_score_matrices", slow):
            self.assert_same_rows(self.harness.run_pipeline(tasks, store, lengths, query, plan, lanes=3), want)

    def test_a_small_search_is_read_in_finer_chunks_than_a_large_one(self):
        plan = self.harness.plan(tile_bytes=14_000, matrix_bytes=40_000)
        path, tasks, store, lengths, query = self.harness.build(self.directory.name, seed=15)
        total = sum(store.source_bytes)
        seen = []
        original = engine.plan_upload_chunks

        def spy(indices, nbytes, chunk_bytes=engine.FIXED_QUERY_CHUNK_BYTES, **kwargs):
            seen.append(chunk_bytes)
            return original(indices, nbytes, chunk_bytes, **kwargs)

        want = self.harness.reference(tasks, store, lengths, query, plan)
        for minimum, per_run, expected in (
            (engine.FIXED_QUERY_MIN_CHUNK_BYTES, 16, engine.FIXED_QUERY_MIN_CHUNK_BYTES),  # too small for 16 chunks
            (1, 4, total // 4),
            (1, 10 ** 9, 1),
        ):
            seen.clear()
            with self.subTest(minimum=minimum, per_run=per_run),                     mock.patch.object(engine, "plan_upload_chunks", spy),                     mock.patch.object(engine, "FIXED_QUERY_MIN_CHUNK_BYTES", minimum),                     mock.patch.object(engine, "FIXED_QUERY_CHUNKS_PER_RUN", per_run):
                self.assert_same_rows(self.harness.run_pipeline(tasks, store, lengths, query, plan), want)
                self.assertEqual(set(seen), {expected})

    def test_the_order_of_the_tasks_does_not_matter(self):
        plan = self.harness.plan(tile_bytes=14_000, matrix_bytes=40_000)
        path, tasks, store, lengths, query = self.harness.build(self.directory.name, seed=9)
        want = self.harness.reference(tasks, store, lengths, query, plan)
        shuffled = list(tasks)
        np.random.default_rng(1).shuffle(shuffled)
        self.assert_same_rows(self.harness.run_pipeline(shuffled, store, lengths, query, plan), want)

    def test_progress_counts_every_target_once(self):
        plan = self.harness.plan(tile_bytes=14_000, matrix_bytes=40_000)
        path, tasks, store, lengths, query = self.harness.build(self.directory.name, seed=10)
        progress = mock.Mock()
        self.harness.run_pipeline(tasks, store, lengths, query, plan, progress=progress)
        self.assertEqual(sum(call.args[0] for call in progress.update.call_args_list), len(tasks))

    def test_a_timed_run_reports_its_phases(self):
        plan = self.harness.plan(tile_bytes=14_000, matrix_bytes=40_000)
        path, tasks, store, lengths, query = self.harness.build(self.directory.name, seed=11)
        timing = SearchTiming()
        self.harness.run_pipeline(tasks, store, lengths, query, plan, timing=timing)
        self.assertGreater(timing.processing, 0)
        self.assertGreaterEqual(timing.setup, 0)
        self.assertGreaterEqual(timing.shutdown, 0)

    def reader_threads(self):
        return [thread for thread in threading.enumerate() if thread.name == "search-embedding-reader"]

    def test_a_missing_embedding_stops_the_search_and_the_reader(self):
        plan = self.harness.plan(tile_bytes=14_000, matrix_bytes=40_000)
        path, tasks, store, lengths, query = self.harness.build(self.directory.name, seed=12)
        store.headers[30] = "absent"
        with self.assertRaises(KeyError):
            self.harness.run_pipeline(tasks, store, lengths, query, plan)
        self.assertEqual(self.reader_threads(), [])

    def test_a_failing_alignment_stops_the_search_and_the_reader(self):
        plan = self.harness.plan(tile_bytes=14_000, matrix_bytes=40_000)
        path, tasks, store, lengths, query = self.harness.build(self.directory.name, seed=13)
        calls = []

        def failing(batch_tasks, target_lengths, matrices):
            calls.append(1)
            if len(calls) == 3:
                raise FloatingPointError("non-finite")
            return ssearch.finish_search_batch(batch_tasks, target_lengths, matrices)

        with mock.patch.object(engine, "cuda_memory_plan", return_value=plan), \
                self.assertRaises(FloatingPointError):
            engine.run_fixed_query_cuda_pipeline(
                tasks, query_embedding=query, store=store, lengths=lengths, device=torch.device("cuda:0"),
                workers=2, lanes=2, batch_alignment_callback=failing,
            )
        self.assertEqual(self.reader_threads(), [])
        # The next search starts clean: nothing is left in the device cache's way.
        self.assert_same_rows(
            self.harness.run_pipeline(tasks, store, lengths, query, plan),
            self.harness.reference(tasks, store, lengths, query, plan),
        )

    def test_the_search_gives_back_its_device_buffers(self):
        # The packed block is about 100 MB here. cuBLAS keeps a few MB for each new stream; nothing else stays.
        harness = FixedQueryHarness()
        harness.FEATURES = 2048
        path, tasks, store, lengths, query = harness.build(
            self.directory.name, targets=48, seed=14, length_range=(150, 250)
        )
        plan = harness.plan(tile_bytes=10 ** 10, matrix_bytes=10 ** 9)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        rows = harness.run_pipeline(tasks, store, lengths, query, plan)
        self.assertEqual(len(rows), 48)
        self.assertLess(torch.cuda.memory_allocated() - before[0], 24 * 1024 ** 2)
        self.assertLess(torch.cuda.memory_reserved() - before[1], 48 * 1024 ** 2)


class SearchPlanUsesTheBatchCallbackTests(unittest.TestCase):
    def test_the_tiled_plan_aligns_whole_microbatches(self):
        candidate = SimpleNamespace(is_cpu=False, device=torch.device("cuda:0"), spec="cuda:0")
        with mock.patch.object(ssearch, "run_fixed_query_cuda_pipeline", return_value=[]) as run:
            ssearch._execute_search_plan(
                (candidate, "tiled", "float32", 2), [(0, "a", "a", np.ones((3, 4), np.float32), "local", -2.0,
                                                      "longer_sequence", "ACD", "ACD")],
                2, "unused.h5", mock.Mock(), [3], np.ones((3, 4), np.float32), False,
            )
        kwargs = run.call_args.kwargs
        self.assertIs(kwargs["batch_alignment_callback"], ssearch.finish_search_batch)
        self.assertNotIn("alignment_callback", kwargs)


if __name__ == "__main__":
    unittest.main()
