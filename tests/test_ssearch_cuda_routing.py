import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np
import h5py
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (
    os.path.join(PROJECT_ROOT, "src"),
    os.path.join(PROJECT_ROOT, "src", "utilities"),
    os.path.join(PROJECT_ROOT, "src", "tools"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Embedding_SSEARCH as ssearch
    from utilities import Embedding_Alignment_Engine as alignment_engine


def make_tasks(count):
    query = np.ones((3, 4), dtype=np.float32)
    return [
        (index, f"h{index}", f"h{index}", query, "local", -2.0, "longer_sequence")
        for index in range(count)
    ]


class SsearchCudaRoutingTests(unittest.TestCase):
    def test_fixed_query_vram_estimate_accounts_for_query_and_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            with h5py.File(path, "w") as hf:
                group = hf.create_group("embeddings")
                group.create_dataset("a", data=np.ones((3, 4), np.float16))
                group.create_dataset("b", data=np.ones((7, 4), np.float32))
            store = alignment_engine.EmbeddingTileStore(path, ["a", "b"], 0)
            estimate = alignment_engine.estimate_fixed_query_cuda_working_set(
                [(0, "a"), (1, "b")],
                query_embedding=np.ones((5, 4), np.float32),
                store=store,
                lengths=[3, 7],
                device=torch.device("cuda"),
                lanes=2,
                memory_info=(12 * 1024 ** 3, 16 * 1024 ** 3),
            )
        self.assertTrue(estimate.feasible)
        self.assertGreater(estimate.tile_bytes, 5 * 4 * 4)
        self.assertGreater(estimate.largest_microbatch_bytes, 0)

    def test_rocm_is_not_reported_as_nvidia_tf32(self):
        with mock.patch.object(torch.version, "cuda", "12.8"), \
                mock.patch.object(torch.version, "hip", "6.2"), \
                mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertFalse(alignment_engine.is_nvidia_cuda(torch.device("cuda")))

    def test_small_database_retains_scalar_cpu_path(self):
        tasks = make_tasks(511)
        cpu = ssearch.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        expected = [{"index": 1}]
        with mock.patch.object(ssearch, "DEVICE_SELECTION", "cpu"), \
                mock.patch.object(
                    ssearch.Hardware_Utils, "get_available_devices", return_value=[cpu]
                ), mock.patch.object(
                    ssearch.Hardware_Utils, "resolve_device_selection", return_value=cpu
                ), mock.patch.object(
                    ssearch, "_run_cpu_search", return_value=expected
                ) as scalar, mock.patch.object(
                    ssearch, "EmbeddingTileStore"
                ) as tile_store:
            actual = ssearch.process_search_tasks(tasks, 2, "unused.h5")
        self.assertIs(actual, expected)
        scalar.assert_called_once()
        tile_store.assert_not_called()

    def test_large_database_uses_adaptive_plan_selector(self):
        tasks = make_tasks(512)
        cpu = ssearch.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        store = mock.Mock()
        store.shapes = [(3, 4)] * len(tasks)
        plan = (cpu, "scalar", "float32", 1)
        expected = [{"index": 2}]
        with mock.patch.object(
            ssearch, "EmbeddingTileStore", return_value=store
        ), mock.patch.object(
            ssearch, "_select_search_plans", return_value=[plan]
        ) as selector, mock.patch.object(
            ssearch, "_execute_search_plan", return_value=expected
        ) as execute:
            actual = ssearch.process_search_tasks(tasks, 2, "unused.h5")
        self.assertIs(actual, expected)
        selector.assert_called_once()
        execute.assert_called_once()

    def test_cost_stratified_sample_obeys_bounds(self):
        tasks = make_tasks(10000)
        lengths = [(index % 100) + 1 for index in range(len(tasks))]
        sample = ssearch._cost_stratified_search_sample(tasks, lengths)
        self.assertEqual(len(sample), 100)
        self.assertEqual(
            len(ssearch._cost_stratified_search_sample(tasks[:512], lengths[:512])),
            16,
        )
        self.assertEqual(
            len(ssearch._cost_stratified_search_sample(tasks, lengths)[:]),
            min(256, max(16, int(len(tasks) * 0.01))),
        )

    def test_tf32_comparison_requires_identical_lengths_and_finite_scores(self):
        baseline = [{
            "index": 0, "raw_score": 20.0, "norm_score": 2.0, "aln_len": 10
        }]
        close = [{
            "index": 0, "raw_score": 20.005, "norm_score": 2.0005, "aln_len": 10
        }]
        changed = [{
            "index": 0, "raw_score": 20.0, "norm_score": 2.0, "aln_len": 9
        }]
        self.assertTrue(ssearch._search_results_equivalent(baseline, close)[0])
        self.assertFalse(ssearch._search_results_equivalent(baseline, changed)[0])


if __name__ == "__main__":
    unittest.main()
