"""Align_Similarity_Matrix as a library (utilities.Align_Library, align_pairs_in_memory):
the import guard, what the call sets and puts back, the planner's options and the scores."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TOOLS_DIR = os.path.join(SRC_DIR, "tools")
for path in (SRC_DIR, TOOLS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from utilities import Align_Library  # noqa: E402
from utilities import Hardware_Acceleration as hardware  # noqa: E402

align = Align_Library.load()

LOCAL_GAP, GLOBAL_GAP = -3.0, -0.5  # not the tool's defaults, and unequal, so a swap shows
THREAD_VARIABLES = align.THREAD_VARIABLES


def write_embeddings(path, count=8, features=12, seed=1):
    rng = np.random.default_rng(seed)
    names = [f"s{k}" for k in range(count)]
    with h5py.File(path, "w") as hf:
        group = hf.create_group("embeddings")
        for name in names:
            group.create_dataset(name, data=rng.standard_normal((int(rng.integers(9, 30)), features)).astype(np.float32))
    return names


def all_pairs(count):
    return np.array([(i, j) for i in range(count) for j in range(i + 1, count)], dtype=np.int64)


def environment():
    return {name: os.environ.get(name) for name in (*THREAD_VARIABLES, align.LIBRARY_VARIABLE)}


class LibraryTestCase(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.embeddings = os.path.join(temp.name, "embeddings.h5")
        self.names = write_embeddings(self.embeddings)
        self.temp = temp.name

    def call(self, pairs=None, **options):
        pairs = all_pairs(len(self.names)) if pairs is None else pairs
        options.setdefault("device_selection", "cpu")
        options.setdefault("workers", 2)
        with mock.patch("builtins.print"):
            return align.align_pairs_in_memory(
                pairs, self.embeddings, self.names, [LOCAL_GAP, GLOBAL_GAP], "ieee_fp32", **options)


class ImportGuardTests(unittest.TestCase):
    def import_in_a_process(self, library, settings_path):
        code = (
            "import json, os, sys\n"
            f"sys.path[:0] = [{SRC_DIR!r}, {TOOLS_DIR!r}]\n"
            "from tools.tool_helpers import Tool_Pipeline\n"
            f"Tool_Pipeline.inherited_settings_path = lambda path: {settings_path!r}\n"
            "import Align_Similarity_Matrix as A\n"
            "print(json.dumps({'omp': os.environ.get('OMP_NUM_THREADS'), 'gap': A.GLOBAL_GAP_P}))\n"
        )
        env = {k: v for k, v in os.environ.items() if k not in THREAD_VARIABLES and k != align.LIBRARY_VARIABLE}
        if library:
            env[align.LIBRARY_VARIABLE] = "1"
        done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        return json.loads(done.stdout.strip().splitlines()[-1])

    def test_a_library_import_leaves_the_threads_and_the_saved_settings_alone(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = os.path.join(temp, "tools_settings.json")
            with open(settings, "w", encoding="utf-8") as handle:
                json.dump({"Align_Similarity_Matrix.py": {"GLOBAL_GAP_P": -7.5}}, handle)
            as_tool = self.import_in_a_process(False, settings)
            as_library = self.import_in_a_process(True, settings)
        self.assertEqual(as_tool, {"omp": "1", "gap": -7.5})
        self.assertEqual(as_library, {"omp": None, "gap": 0.0})


class EntryTests(LibraryTestCase):
    def test_the_scores_are_the_tools_own_with_the_networks_gap_penalties_in_the_pairs_order(self):
        pairs = all_pairs(6)[::-1].copy()  # not in the order a loop would give
        got = self.call(pairs=pairs)
        cpu = torch.device("cpu")
        with h5py.File(self.embeddings, "r") as hf:
            for k, (i, j) in enumerate(pairs.tolist()):
                matrix = align.compute_score_matrix_torch(
                    hf["embeddings"][self.names[i]][:], hf["embeddings"][self.names[j]][:], cpu)
                g_raw, g_len, l_raw, l_len = align.global_local_pair_scores(matrix, GLOBAL_GAP, LOCAL_GAP)
                self.assertEqual((got.l_score[k], got.l_len[k], got.g_score[k], got.g_len[k]),
                                 (np.float32(l_raw), l_len, np.float32(g_raw), g_len), f"pair {i}-{j}")
        self.assertEqual((got.precision, got.timed, got.trial_pairs), ("ieee_fp32", False, 0))
        self.assertIn("CPU", got.device)

    def test_the_call_works_while_the_caller_holds_the_embeddings_file_open_the_shared_way(self):
        # Embedding MSA keeps its embeddings open (opened by open_embeddings_shared) during the call.
        from utilities.HDF5_Storage import open_embeddings_shared

        with open_embeddings_shared(self.embeddings):
            got = self.call(pairs=all_pairs(4))
        self.assertEqual(len(got.l_score), 6)

    def test_the_module_values_and_the_environment_are_put_back_also_after_an_error(self):
        before_values = {name: getattr(align, name) for name in align._LIBRARY_GLOBALS}
        before_environment = environment()
        seen = {}

        def fail(*args, **kwargs):
            seen.update(gaps=(align.LOCAL_GAP_P, align.GLOBAL_GAP_P), device=align.DEVICE_SELECTION,
                        environment=environment())
            raise RuntimeError("boom")

        with mock.patch.object(align, "process_cpu_tasks", fail):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                self.call()
        self.assertEqual(seen["gaps"], (LOCAL_GAP, GLOBAL_GAP))
        self.assertEqual(seen["device"], "cpu")
        self.assertEqual(seen["environment"][align.LIBRARY_VARIABLE], "1")
        self.assertEqual({k: seen["environment"][k] for k in THREAD_VARIABLES}, {k: "1" for k in THREAD_VARIABLES})
        self.assertEqual({name: getattr(align, name) for name in align._LIBRARY_GLOBALS}, before_values)
        self.assertEqual(environment(), before_environment)
        self.call(pairs=all_pairs(3))
        self.assertEqual(environment(), before_environment)

    def test_the_pool_children_are_single_threaded_and_take_the_gap_penalties(self):
        import multiprocessing
        children = {}
        with align._library_state(_library_pool_context=multiprocessing.get_context("spawn"),
                                  _library_worker_settings=((LOCAL_GAP, GLOBAL_GAP), True)):
            with align._library_pool_context.Pool(
                    processes=1, initializer=align.init_worker,
                    initargs=(self.embeddings, align._library_worker_settings)) as pool:
                children["torch"] = pool.apply(torch.get_num_threads)
                children["omp"] = pool.apply(os.getenv, ("OMP_NUM_THREADS",))
        self.assertEqual(children, {"torch": 1, "omp": "1"})

    def test_the_initializer_limits_a_child_to_one_thread_even_when_torch_is_already_running_more(self):
        # A child that has already loaded torch with several threads (the environment came too late) must still end with one.
        saved = {name: getattr(align, name) for name in ("LOCAL_GAP_P", "GLOBAL_GAP_P", "worker_hf", "worker_device")}
        before = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, before)
        for name, value in saved.items():
            self.addCleanup(setattr, align, name, value)
        torch.set_num_threads(max(2, before))
        self.assertGreater(torch.get_num_threads(), 1)
        align.init_worker(self.embeddings, ((LOCAL_GAP, GLOBAL_GAP), True))
        self.addCleanup(align.worker_hf.close)
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual((align.LOCAL_GAP_P, align.GLOBAL_GAP_P), (LOCAL_GAP, GLOBAL_GAP))


class FakeAcceleratorTests(LibraryTestCase):
    """The planner's options, with the planner and the batch runner replaced by recorders."""

    def setUp(self):
        super().setUp()
        cpu = hardware.DeviceCandidate("cpu", "CPU", torch.device("cpu"), "cpu")
        gpu = hardware.DeviceCandidate("cuda:0", "Fake GPU", torch.device("cpu"), "cuda", 0, True)
        self.cpu, self.gpu = cpu, gpu
        self.timed, self.run = [], []
        patches = [
            mock.patch.object(align.Hardware_Utils, "get_available_devices", lambda **_: [cpu, gpu]),
            mock.patch.object(align, "_benchmark_processing_plans", self.fake_benchmark),
            mock.patch.object(align, "process_batch", self.fake_process_batch),
            mock.patch.object(align, "_first_listed_plan", lambda candidate, *a: hardware.BenchmarkResult(
                candidate, 0.0, variant="scalar")),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def rows(tasks):
        return [(i, j, 1.0, 2, 3.0, 4) for i, j, *_ in tasks]

    def fake_benchmark(self, sample, workers, h5, batch_id, **kwargs):
        self.timed.append(list(sample))
        winner = hardware.BenchmarkResult(self.cpu, 10.0, variant="scalar")
        kwargs["result_sink"][id(winner)] = self.rows(sample)[:7]
        return [winner]

    def fake_process_batch(self, tasks, *args, result_writer=None, **kwargs):
        self.run.append(list(tasks))
        result_writer(self.rows(tasks))

    def test_two_usable_devices_time_the_plans_on_a_fraction_and_keep_those_pairs(self):
        pairs = np.array([(i, j) for i in range(40) for j in range(i + 1, 40)])[:400]
        self.names = [f"s{k}" for k in range(40)]
        write_embeddings(self.embeddings, count=40)
        got = self.call(pairs=pairs, device_selection="auto")
        self.assertEqual(len(self.timed), 1)
        self.assertEqual(len(self.timed[0]), 32)  # 5% of 400 is 20: the warm-up's 32 is more
        self.assertEqual(got.trial_pairs, 7)
        self.assertEqual(sum(len(batch) for batch in self.run), 400 - 7)
        self.assertTrue(got.timed)

    def test_one_usable_device_or_a_named_one_is_not_timed(self):
        with mock.patch.object(align.Hardware_Utils, "get_available_devices", lambda **_: [self.cpu]):
            self.call(device_selection="auto")
        self.call(device_selection="cuda:0")
        self.call(device_selection="cpu")
        self.assertEqual(self.timed, [])
        self.assertEqual(len(self.run), 3)

    def test_the_trial_is_a_fraction_of_the_pairs_never_fewer_than_the_warm_up_needs(self):
        def tasks(count):
            return [(k, k + 1, "a", "b") for k in range(count)]

        lengths = list(range(1, 100_003))
        for count, workers, expected in ((5_000, 8, 250), (100_000, 8, 5_000), (400, 8, 32), (20, 8, 20),
                                         (1_000, 40, 80)):
            with self.subTest(count=count):
                sample = align._trial_sample(tasks(count), lengths, workers, 0.05)
                self.assertEqual(len(sample), expected)
                self.assertEqual(len({(t[0], t[1]) for t in sample}), expected)

    def test_a_precision_no_usable_device_can_do_is_aligned_in_fp32_with_a_note(self):
        for requested, text in (("tf32", "TF32"), ("bf16", "BF16")):
            with self.subTest(requested=requested):
                with mock.patch("builtins.print"):
                    got = align.align_pairs_in_memory(
                        all_pairs(4), self.embeddings, self.names, [LOCAL_GAP, GLOBAL_GAP], requested,
                        device_selection="cpu", workers=1)
                self.assertEqual((got.precision, got.requested_precision), ("ieee_fp32", requested))
                self.assertTrue(any(text in note for note in got.notes), got.notes)


if __name__ == "__main__":
    unittest.main()
