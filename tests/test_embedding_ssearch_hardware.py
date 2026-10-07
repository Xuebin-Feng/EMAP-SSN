"""Embedding_SSEARCH hardware plans: device routing, benchmarks and precision checks.

Covers process_search_tasks and _select_search_plans, the SearchSelector and
SearchPlan benchmark helpers they use from Tool_Pipeline, the fixed-query CUDA
working-set estimate and the TF32/BF16 result comparisons. Accelerators are
simulated; the real searches run on the CPU.
"""

import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
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
    import Embedding_Alignment_Engine as alignment_engine
from tools.tool_helpers import Tool_Pipeline as tool_pipeline  # noqa: E402
from tools.tool_helpers.Tool_Pipeline import (  # noqa: E402
    SearchPlan,
    SearchSelector,
    SearchTiming,
    rank_plans,
    stratified,
)


def make_tasks(count):
    query = np.ones((3, 4), dtype=np.float32)
    return [
        (index, f"h{index}", f"h{index}", query, "local", -2.0, "longer_sequence",
         "ACD", "ACD")
        for index in range(count)
    ]


def cpu():
    return ssearch.Hardware_Utils.DeviceCandidate("cpu", "CPU", torch.device("cpu"), "cpu")


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
        # is_nvidia_cuda also requires torch.cuda.is_available(). Reporting it
        # as available leaves the HIP version as the only reason to refuse, on
        # machines with or without a GPU, and never queries the driver.
        with mock.patch.object(torch.version, "cuda", "12.8"), \
                mock.patch.object(torch.version, "hip", "6.2"), \
                mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertFalse(alignment_engine.is_nvidia_cuda(torch.device("cuda")))
        with mock.patch.object(torch.version, "cuda", "12.8"), \
                mock.patch.object(torch.version, "hip", None), \
                mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertTrue(alignment_engine.is_nvidia_cuda(torch.device("cuda")))
            self.assertFalse(alignment_engine.is_nvidia_cuda(torch.device("cpu")))

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
        # 0.02 over 10 aligned residues is a drift of 0.002 per residue.
        drifted = [dict(baseline[0], raw_score=20.02)]
        self.assertEqual(
            ssearch._search_results_equivalent(baseline, drifted),
            (False, "per-aligned-residue drift exceeded 0.001"),
        )
        for field in ("raw_score", "norm_score"):
            for value in (np.nan, np.inf):
                with self.subTest(field=field, value=value):
                    candidate = [dict(baseline[0], **{field: value})]
                    self.assertEqual(
                        ssearch._search_results_equivalent(baseline, candidate),
                        (False, "non-finite candidate result for target 0"),
                    )
                    self.assertFalse(
                        ssearch._search_results_equivalent(candidate, close)[0]
                    )

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


class SelectorTests(unittest.TestCase):
    def selector(self, seconds=0.01, count=1000):
        self.now = 0.0
        def execute(plan, tasks, timing):
            self.now += seconds
            timing.setup, timing.processing, timing.shutdown = 0.001, seconds, 0.001
            return [{"index": task[0]} for task in tasks]
        return SearchSelector(make_tasks(count), [3] * count, 3, execute, clock=lambda: self.now)

    def test_budget_is_soft_but_stops_new_trials(self):
        selector = self.selector(seconds=6)
        plan = SearchPlan(cpu(), "serial", "float32")
        with redirect_stdout(io.StringIO()):
            self.assertTrue(selector.trial(plan, required=True))
            self.assertFalse(selector.trial(plan))
        self.assertEqual(len(plan.results), 16)
        self.assertEqual(len(plan.observations), 1)

    def test_tiny_baseline_retains_real_work(self):
        selector = self.selector(count=20)
        plan = SearchPlan(cpu(), "serial", "float32")
        with redirect_stdout(io.StringIO()):
            self.assertTrue(selector.baseline(plan))
        self.assertEqual(len(plan.results), 4)
        self.assertEqual(selector.budget, 0.5)

    def test_short_budget_does_not_assume_pool_has_serial_startup(self):
        selector = self.selector(count=10000)
        serial = SearchPlan(cpu(), "serial", "float32")
        with redirect_stdout(io.StringIO()):
            selector.trial(serial, required=True)
            selector.budget = 0.5
            self.assertFalse(selector.allowed(SearchPlan(cpu(), "pool", "float32")))

    def test_repeated_targets_are_deduplicated(self):
        selector = self.selector()
        plan = SearchPlan(cpu(), "serial", "float32")
        with redirect_stdout(io.StringIO()):
            selector.trial(plan, required=True)
            selector.trial(plan, required=True)
        self.assertEqual(len(plan.results), 16)
        self.assertEqual(len(plan.observations), 2)

    def test_cost_prediction_scales_with_remaining_work(self):
        plan = SearchPlan(cpu(), "serial", "float32")
        timing = SearchTiming()
        timing.setup, timing.shutdown = 1, 0.5
        plan.observations = [(timing, 0.1)]
        self.assertAlmostEqual(plan.predicted({0: 10, 1: 30}), 5.5)
        plan.results[0] = {"index": 0}
        self.assertAlmostEqual(plan.predicted({0: 10, 1: 30}), 4.5)

    def test_ties_prefer_serial_then_fewer_lanes(self):
        serial = SearchPlan(cpu(), "serial", "float32")
        pool = SearchPlan(cpu(), "pool", "float32")
        for plan, rate in ((serial, 1.04), (pool, 1.0)):
            plan.observations = [(SearchTiming(), rate)]
        self.assertIs(rank_plans([pool, serial], {0: 1})[0], serial)

    def test_sample_contains_extremes_and_no_duplicates(self):
        tasks = make_tasks(100)
        sample = stratified(tasks, list(range(100)), 16)
        self.assertEqual((sample[0][0], sample[-1][0]), (0, 99))
        self.assertEqual(len({task[0] for task in sample}), 16)

    def test_failure_does_not_retain_partial_trial(self):
        selector = self.selector()
        selector.execute = lambda *args: [{"index": 0}]
        plan = SearchPlan(cpu(), "serial", "float32")
        with redirect_stdout(io.StringIO()):
            self.assertFalse(selector.trial(plan, required=True))
        self.assertEqual(plan.results, {})

    def test_failover_reuses_only_replacement_results(self):
        tasks = make_tasks(3)
        first = SearchPlan(cpu(), "pool", "float32")
        second = SearchPlan(cpu(), "serial", "float32")
        first.results = {0: {"index": 0, "source": "failed"}}
        second.results = {1: {"index": 1, "source": "replacement"}}
        def execute(plan, pending, *args):
            if plan[1] == "pool":
                raise RuntimeError("test failure")
            self.assertEqual([task[0] for task in pending], [0, 2])
            return [{"index": task[0], "source": "replacement"} for task in pending]
        with mock.patch.object(ssearch, "EmbeddingTileStore", return_value=mock.Mock(shapes=[(3, 4)] * 3)), \
                mock.patch.object(ssearch, "DEVICE_SELECTION", "auto"), \
                mock.patch.object(ssearch, "_select_search_plans", return_value=[first, second]), \
                mock.patch.object(ssearch, "_execute_search_plan", side_effect=execute), redirect_stdout(io.StringIO()):
            results = ssearch.process_search_tasks(tasks, 2, "unused.h5")
        self.assertTrue(all(row["source"] == "replacement" for row in results))


class RealCpuTests(unittest.TestCase):
    def test_serial_and_pool_match_all_normalizations(self):
        rng = np.random.default_rng(412)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "embeddings.h5")
            lengths = (8, 12, 17)
            with h5py.File(path, "w") as hf:
                group = hf.create_group("embeddings")
                for index, length in enumerate(lengths):
                    group.create_dataset(f"h{index}", data=rng.normal(size=(length, 8)).astype(np.float32))
            query = rng.normal(size=(10, 8)).astype(np.float32)
            residues = np.array(list("ACDX"))
            query_seq = "".join(rng.choice(residues, size=10))
            target_seqs = ["".join(rng.choice(residues, size=length)) for length in lengths]
            tasks = []
            # Unique result identities for each mode/normalization combination
            # that SSEARCH accepts: main() refuses local alignment_length.
            for mode in ("local", "global"):
                for norm in ("alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"):
                    if mode == "local" and norm == "alignment_length":
                        continue
                    for index in range(3):
                        tasks.append((len(tasks), f"h{index}", f"h{index}", query, mode, -2.0, norm,
                                      query_seq, target_seqs[index]))
            serial_timing, pool_timing = SearchTiming(), SearchTiming()
            serial = ssearch._run_serial_search(tasks, path, timing=serial_timing)
            pool = ssearch._run_cpu_search(tasks, 2, path, False, timing=pool_timing)
            self.assertEqual(serial, sorted(pool, key=lambda row: row["index"]))
            self.assertTrue(all(0.0 <= row["identity"] <= 100.0 for row in serial))
            self.assertGreater(serial_timing.processing, 0)
            self.assertGreater(pool_timing.processing, 0)
            self.assertGreater(pool_timing.setup, 0)


class FakeClockTime:
    """The time module, except that perf_counter reads a simulated clock.

    Plan selection reads perf_counter only in Embedding_SSEARCH and in the
    Tool_Pipeline selector and timing helpers, so the simulated clock replaces
    the `time` name in those two modules instead of time.perf_counter itself,
    which every other module and thread in the process shares.
    """

    def __init__(self, clock):
        self.perf_counter = clock

    def __getattr__(self, name):
        return getattr(time, name)


class PlanSelectionTests(unittest.TestCase):
    def run_selector(self, selection="auto", lanes="auto", precision="float32", infeasible=False,
                     fail=False, tf32_changed=False):
        gpu = ssearch.Hardware_Utils.DeviceCandidate("cuda:0", "GPU", torch.device("cuda:0"), "cuda")
        other = ssearch.Hardware_Utils.DeviceCandidate("xpu:0", "XPU", torch.device("xpu:0"), "xpu")
        devices = [cpu(), gpu, other]
        tasks = make_tasks(10000)
        now = [0.0]
        calls = []
        def execute(plan, selected, *args, timing=None):
            calls.append(plan)
            if fail:
                raise RuntimeError("unavailable")
            candidate, variant, prec, count = plan
            # A 100-second CPU workload makes refinements worthwhile.
            rate = (0.01 if variant == "serial" else 0.002) / count
            if variant == "tiled":
                rate *= 0.5
            if prec == "tf32":
                rate *= 0.5
            duration = len(selected) * rate
            now[0] += duration + 0.002
            if timing is not None:
                timing.setup, timing.processing, timing.shutdown = 0.001, duration, 0.001
            return [{"index": t[0], "raw_score": 1.0, "norm_score": 1.0,
                     "aln_len": 2 if tf32_changed and prec == "tf32" else 1} for t in selected]
        with ExitStack() as stack:
            for name, value in (("DEVICE_SELECTION", selection), ("ACCELERATOR_LANES", lanes),
                                ("ACCELERATOR_PRECISION", precision)):
                stack.enter_context(mock.patch.object(ssearch, name, value))
            clock = FakeClockTime(lambda: now[0])
            stack.enter_context(mock.patch.object(ssearch, "time", clock))
            stack.enter_context(mock.patch.object(tool_pipeline, "time", clock))
            stack.enter_context(mock.patch.object(ssearch.Hardware_Utils, "get_available_devices", return_value=devices))
            stack.enter_context(mock.patch.object(ssearch.Hardware_Utils, "release_device_cache"))
            stack.enter_context(mock.patch.object(ssearch, "is_nvidia_cuda", side_effect=lambda d: d.type == "cuda"))
            stack.enter_context(mock.patch.object(ssearch, "estimate_fixed_query_cuda_working_set",
                                                  return_value=mock.Mock(feasible=not infeasible, reason="too large")))
            stack.enter_context(mock.patch.object(ssearch, "_execute_search_plan", side_effect=execute))
            stack.enter_context(redirect_stdout(io.StringIO()))
            plans = ssearch._select_search_plans(tasks, 4, "unused.h5",
                         mock.Mock(feature_dimension=4, dtypes=[np.dtype("float32")]), [3] * len(tasks), tasks[0][3])
        return plans, calls

    def test_explicit_device_and_lanes_are_honored(self):
        plans, calls = self.run_selector(selection="cuda:0", lanes=3)
        self.assertTrue(plans)
        self.assertTrue(all(p[0].spec == "cuda:0" and p[3] == 3 for p in calls))

    def test_multiple_devices_and_repeat_limit(self):
        plans, calls = self.run_selector()
        self.assertTrue({"cpu", "cuda:0", "xpu:0"}.issubset({p[0].spec for p in calls}))
        self.assertTrue(all(len(p.observations) <= 3 for p in plans))
        self.assertTrue(all(p.lanes <= 4 for p in plans))

    def test_tiled_memory_failure_skips_execution(self):
        plans, calls = self.run_selector(infeasible=True)
        self.assertTrue(plans)
        self.assertTrue(all(p[1] != "tiled" for p in calls))

    def test_forced_precision_never_runs_cpu(self):
        plans, calls = self.run_selector(precision="tf32")
        self.assertTrue(plans)
        self.assertTrue(all(p[2] == "tf32" and not p[0].is_cpu for p in calls))

    def test_manual_failure_does_not_fallback_to_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "No SSEARCH hardware plan"):
            self.run_selector(selection="cuda:0", fail=True)

    def test_auto_tf32_requires_equivalence(self):
        plans, calls = self.run_selector(precision="automatic_32bit", tf32_changed=True)
        self.assertTrue(any(p[2] == "tf32" for p in calls))
        self.assertTrue(all(p.precision == "float32" for p in plans))

    def test_auto_tf32_can_win_when_equivalent_and_faster(self):
        plans, calls = self.run_selector(precision="automatic_32bit")
        self.assertTrue(any(p.precision == "tf32" for p in plans))


if __name__ == "__main__":
    unittest.main()
