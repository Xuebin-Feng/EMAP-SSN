import io
import tempfile
import unittest
from contextlib import redirect_stdout, ExitStack
from pathlib import Path
from unittest import mock

from tests.test_ssearch_cuda_routing import ssearch, make_tasks
import h5py
import numpy as np
import torch
from utilities.Ssearch_Benchmark import SearchPlan, SearchSelector, SearchTiming, rank_plans, stratified


def cpu():
    return ssearch.Hardware_Utils.DeviceCandidate("cpu", "CPU", torch.device("cpu"), "cpu")


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
            with h5py.File(path, "w") as hf:
                group = hf.create_group("embeddings")
                for index, length in enumerate((8, 12, 17)):
                    group.create_dataset(f"h{index}", data=rng.normal(size=(length, 8)).astype(np.float32))
            query = rng.normal(size=(10, 8)).astype(np.float32)
            tasks = []
            # Unique result identities for each mode/normalization combination.
            for mode in ("local", "global"):
                for norm in ("alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"):
                    for index in range(3):
                        tasks.append((len(tasks), f"h{index}", f"h{index}", query, mode, -2.0, norm))
            serial_timing, pool_timing = SearchTiming(), SearchTiming()
            serial = ssearch._run_serial_search(tasks, path, timing=serial_timing)
            pool = ssearch._run_cpu_search(tasks, 2, path, False, timing=pool_timing)
            self.assertEqual(serial, sorted(pool, key=lambda row: row["index"]))
            self.assertGreater(serial_timing.processing, 0)
            self.assertGreater(pool_timing.processing, 0)
            self.assertGreater(pool_timing.setup, 0)


class RoutingTests(unittest.TestCase):
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
            stack.enter_context(mock.patch.object(ssearch.time, "perf_counter", side_effect=lambda: now[0]))
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
