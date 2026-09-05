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
        self.assertEqual(
            len(ssearch._bf16_search_validation_sample(tasks, lengths)),
            2048,
        )
        self.assertEqual(
            len(
                ssearch._bf16_search_validation_sample(
                    tasks[:512], lengths[:512]
                )
            ),
            512,
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
                store=mock.Mock(),
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
            if len(call.args[1]) == 30
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
        self.assertEqual(len(benchmark_calls), 6)
        self.assertEqual(len(plans), 6)
        self.assertEqual(output.getvalue().count("explicit low-precision BF16"), 1)
        self.assertEqual(output.getvalue().count("BF16 validation report:"), 2)


if __name__ == "__main__":
    unittest.main()
