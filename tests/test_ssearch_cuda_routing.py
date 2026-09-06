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

    def test_empty_search_does_not_inspect_database(self):
        with mock.patch.object(ssearch, "EmbeddingTileStore") as store:
            self.assertEqual(ssearch.process_search_tasks([], 2, "unused.h5"), [])
        store.assert_not_called()

    def test_all_database_sizes_use_adaptive_selector(self):
        cpu = ssearch.Hardware_Utils.DeviceCandidate("cpu", "CPU", torch.device("cpu"), "cpu")
        for count in (1, 511, 512):
            tasks = make_tasks(count)
            store = mock.Mock(shapes=[(3, 4)] * count)
            plan = ssearch.SearchPlan(cpu, "serial", "float32")
            plan.results = {0: {"index": 0}}
            expected = [{"index": index} for index in range(count)]
            with mock.patch.object(ssearch, "EmbeddingTileStore", return_value=store), \
                    mock.patch.object(ssearch, "_select_search_plans", return_value=[plan]) as selector, \
                    mock.patch.object(ssearch, "_execute_search_plan", return_value=expected[1:]):
                self.assertEqual(ssearch.process_search_tasks(tasks, 2, "unused.h5"), expected)
            selector.assert_called_once()

    def test_bf16_sample_obeys_bounds(self):
        tasks = make_tasks(10000)
        lengths = [(index % 100) + 1 for index in range(len(tasks))]
        self.assertEqual(len(ssearch._bf16_search_validation_sample(tasks, lengths)), 2048)
        self.assertEqual(len(ssearch._bf16_search_validation_sample(tasks[:512], lengths)), 512)

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

    @staticmethod
    def _search_validation_rows(count, changed_count):
        baseline = [
            {
                "index": index,
                "raw_score": 100.0,
                "norm_score": 1.0,
                "aln_len": 100,
            }
            for index in range(count)
        ]
        candidate = [dict(result) for result in baseline]
        for index in range(changed_count):
            candidate[index].update(
                raw_score=104.0,
                norm_score=1.0,
                aln_len=104,
            )
        return baseline, candidate

    def test_bf16_ssearch_reports_all_changed_targets_without_rejection(self):
        baseline, candidate = self._search_validation_rows(2048, 2048)
        for result in candidate:
            result.update(raw_score=1000.0, aln_len=300)

        report = ssearch._search_bf16_validation(baseline, candidate)

        self.assertEqual(report.sample_count, 2048)
        self.assertEqual(report.changed_case_count, 2048)
        self.assertEqual(report.modes[0].length_percentage_drift.maximum, 200.0)
        self.assertEqual(report.modes[0].score_percentage_drift.maximum, 900.0)

    def test_bf16_ssearch_preserves_normalized_score_finiteness_check(self):
        baseline, candidate = self._search_validation_rows(33, 1)
        candidate[0]["norm_score"] = np.inf
        with self.assertRaisesRegex(
            alignment_engine.BF16ValidationIntegrityError,
            "non-finite BF16 result",
        ):
            ssearch._search_bf16_validation(baseline, candidate)

    def test_bf16_ssearch_validates_2048_once_per_variant_not_lane(self):
        tasks = make_tasks(3000)
        lengths = [(index % 100) + 1 for index in range(len(tasks))]
        candidate = ssearch.Hardware_Utils.DeviceCandidate(
            "cuda:0", "Test GPU", torch.device("cuda:0"), "cuda"
        )
        estimate = mock.Mock(feasible=True, reason="safe")

        def execute(plan, selected_tasks, *_args, **_kwargs):
            timing = _kwargs.get("timing")
            if timing is not None:
                timing.setup, timing.processing, timing.shutdown = 0.001, 0.1 / plan[3], 0.001
            return [
                {
                    "index": int(task[0]),
                    "raw_score": 100.0,
                    "norm_score": 1.0,
                    "aln_len": 100,
                }
                for task in selected_tasks
            ]

        output = io.StringIO()
        with mock.patch.object(
            ssearch, "ACCELERATOR_PRECISION", "bf16"
        ), mock.patch.object(
            ssearch, "DEVICE_SELECTION", "auto"
        ), mock.patch.object(
            ssearch.Hardware_Utils,
            "get_available_devices",
            return_value=[candidate],
        ), mock.patch.object(
            ssearch.Hardware_Utils,
            "resolve_device_selection",
            return_value=None,
        ), mock.patch.object(
            ssearch.Hardware_Utils, "release_device_cache"
        ), mock.patch.object(
            ssearch, "bf16_accelerator_support", return_value=(True, "supported")
        ), mock.patch.object(
            ssearch, "_lane_candidates", return_value=[1, 2, 4]
        ), mock.patch.object(
            ssearch,
            "estimate_fixed_query_cuda_working_set",
            return_value=estimate,
        ), mock.patch.object(
            ssearch, "_execute_search_plan", side_effect=execute
        ) as run, redirect_stdout(output):
            plans = ssearch._select_search_plans(
                tasks,
                workers=4,
                input_h5="unused.h5",
                store=mock.Mock(dtypes=[np.dtype("float32")]),
                lengths=lengths,
                query_embedding=np.ones((3, 4), np.float32),
            )

        validation_calls = [
            call
            for call in run.call_args_list
            if len(call.args[1]) == 2048
        ]
        benchmark_calls = [
            call
            for call in run.call_args_list
            if len(call.args[1]) == 16
        ]
        self.assertEqual(len(validation_calls), 3)
        self.assertEqual(
            [(call.args[0][1], call.args[0][2], call.args[0][3]) for call in validation_calls],
            [
                ("scalar", "float32", 1),
                ("scalar", "bf16", 1),
                ("tiled", "bf16", 1),
            ],
        )
        self.assertGreaterEqual(len(benchmark_calls), 2)
        self.assertTrue(plans)
        self.assertEqual(output.getvalue().count("explicit low-precision BF16"), 1)
        self.assertEqual(output.getvalue().count("BF16 validation report:"), 2)


if __name__ == "__main__":
    unittest.main()
