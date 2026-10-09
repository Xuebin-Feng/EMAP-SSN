"""Tests for utilities/Benchmark_Record.py, the opt-in record of the Auto hardware
decisions, and for the places that record: rank_benchmark_results and its
callers, the database search's final plan ranking, the alignment's FP32/TF32
choice and the host cache."""

import ast
import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for _path in (SRC_DIR, SRC_DIR / "utilities", SRC_DIR / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from utilities import Benchmark_Record  # noqa: E402
from utilities import Hardware_Acceleration as Hardware_Utils  # noqa: E402
from tools.tool_helpers.Tool_Pipeline import SearchPlan  # noqa: E402

# The tests package points tool imports at a missing settings file.
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as similarity_matrix  # noqa: E402
    import Embedding_SSEARCH as ssearch  # noqa: E402

RECORDING_KEYS = (Benchmark_Record.RECORD_VARIABLE, Benchmark_Record.STAGE_VARIABLE)


def cpu():
    return Hardware_Utils.DeviceCandidate("cpu", "CPU", torch.device("cpu"), "cpu")


def gpu():
    return Hardware_Utils.DeviceCandidate("cuda:0", "Test GPU (CUDA)", torch.device("cuda:0"), "cuda", 0, True)


class RecordingFixture:
    """Point SSN_BENCHMARK_RECORD at a private file for one test."""

    def start_recording(self, stage="3"):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.record_file = Path(folder.name) / "decisions.jsonl"
        patcher = mock.patch.dict(os.environ, {
            Benchmark_Record.RECORD_VARIABLE: str(self.record_file),
            Benchmark_Record.STAGE_VARIABLE: stage,
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def decisions(self):
        return Benchmark_Record.read_records(self.record_file)

    def not_recording(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in RECORDING_KEYS:
            os.environ.pop(key, None)


class RecorderTests(RecordingFixture, unittest.TestCase):
    def results(self):
        return [
            Hardware_Utils.BenchmarkResult(cpu(), 100.0),
            Hardware_Utils.BenchmarkResult(gpu(), 101.0, lanes=2, variant="tiled", peak_memory_bytes=5 << 30,
                                           execution_plan=SimpleNamespace(profile_name="balanced")),
            Hardware_Utils.BenchmarkResult(gpu(), None, lanes=4, error="OutOfMemoryError: no room"),
            Hardware_Utils.BenchmarkResult(gpu(), 140.0, lanes=8),
        ]

    def test_nothing_is_recorded_or_read_without_the_variable(self):
        self.not_recording()
        with tempfile.TemporaryDirectory() as folder:
            previous = os.getcwd()
            os.chdir(folder)
            try:
                self.assertFalse(Benchmark_Record.record("alignment_plan", value=1))
                # Without recording, a ranking's results are never even read.
                self.assertFalse(Benchmark_Record.record_ranking(
                    [object()], [object()], higher_is_better=True, tie_fraction=0.03))
                self.assertFalse(Benchmark_Record.record_search_plans([object()], [object()], {}))
                self.assertFalse(Benchmark_Record.record_host_cache(object(), "auto"))
                ranked = Hardware_Utils.rank_benchmark_results(
                    self.results(), higher_is_better=True, decision={"kind": "alignment_plan", "unit": "pairs/s"})
            finally:
                os.chdir(previous)
            self.assertEqual(os.listdir(folder), [])
        self.assertEqual([result.value for result in ranked], [140.0, 100.0, 101.0])

    def test_a_ranking_is_one_line_with_its_candidates_order_and_winner(self):
        self.start_recording()
        results = self.results()
        ranked = Hardware_Utils.rank_benchmark_results(
            results, higher_is_better=True, decision={"kind": "alignment_plan", "unit": "pairs/s", "pairs": 64})
        [decision] = self.decisions()
        self.assertEqual(self.record_file.read_text(encoding="utf-8").count("\n"), 1)
        self.assertEqual(
            {key: decision[key] for key in ("kind", "stage", "unit", "direction", "tie_fraction", "pairs")},
            {"kind": "alignment_plan", "stage": "3", "unit": "pairs/s", "direction": "higher",
             "tie_fraction": 0.03, "pairs": 64},
        )
        self.assertEqual(decision["candidates"][1], {
            "device": "Test GPU (CUDA)", "spec": "cuda:0", "backend": "cuda", "variant": "tiled", "lanes": 2,
            "value": 101.0, "error": None, "peak_memory_bytes": 5 << 30, "profile": "balanced",
        })
        self.assertEqual(decision["candidates"][2]["error"], "OutOfMemoryError: no room")
        self.assertIsNone(decision["candidates"][2]["value"])
        self.assertEqual(decision["ranking"], [results.index(result) for result in ranked])
        self.assertEqual(decision["ranking"], [3, 0, 1])
        self.assertEqual((decision["winner"], decision["fastest"]), (3, 3))

    def test_the_tie_preference_shows_as_a_winner_that_is_not_the_fastest(self):
        self.start_recording()
        results = self.results()[:3]
        Hardware_Utils.rank_benchmark_results(results, higher_is_better=True, decision={"kind": "alignment_plan"})
        [decision] = self.decisions()
        self.assertEqual((decision["winner"], decision["fastest"]), (0, 1), "CPU wins inside the 3% tie")
        results[0] = Hardware_Utils.BenchmarkResult(cpu(), 102.0)
        Hardware_Utils.rank_benchmark_results(results, higher_is_better=False, decision={"kind": "msa_device"})
        lower = self.decisions()[1]
        self.assertEqual((lower["direction"], lower["winner"], lower["fastest"]), ("lower", 0, 1),
                         "the CPU's 102 s ties with the GPU's 101 s")

    def test_recording_never_changes_a_ranking(self):
        for higher_is_better in (True, False):
            with self.subTest(higher_is_better=higher_is_better):
                results = self.results()
                self.not_recording()
                quiet = Hardware_Utils.rank_benchmark_results(results, higher_is_better=higher_is_better)
                self.start_recording()
                recorded = Hardware_Utils.rank_benchmark_results(
                    results, higher_is_better=higher_is_better, decision={"kind": "layout_device", "unit": "s"})
                self.assertEqual([id(result) for result in recorded], [id(result) for result in quiet])

    def test_a_generator_of_results_is_ranked_and_recorded_whole(self):
        self.start_recording()
        ranked = Hardware_Utils.rank_benchmark_results(
            (result for result in self.results()), higher_is_better=True, decision={"kind": "injection_plan"})
        self.assertEqual(len(ranked), 3)
        self.assertEqual(len(self.decisions()[0]["candidates"]), 4)

    def test_values_are_written_as_plain_json(self):
        self.start_recording()
        self.assertTrue(Benchmark_Record.record(
            "matmul_precision", rate=np.float32(1.5), missing=float("nan"), infinite=float("inf"),
            count=np.int64(3), flag=np.bool_(True), shape=(2, 3), plan=object(),
        ))
        line = self.record_file.read_text(encoding="utf-8")
        self.assertNotIn("NaN", line)
        [decision] = self.decisions()
        self.assertEqual(
            (decision["rate"], decision["missing"], decision["infinite"], decision["count"], decision["flag"],
             decision["shape"]),
            (1.5, None, None, 3, True, [2, 3]),
        )
        self.assertIsInstance(decision["plan"], str)
        self.assertIsInstance(decision["time"], float)

    def test_text_outside_ascii_is_kept_as_utf_8(self):
        self.start_recording()
        Benchmark_Record.record("embedding_device", device="Radeon™ RX 7900 — 显卡")
        self.assertIn("显卡", self.record_file.read_bytes().decode("utf-8"))
        self.assertEqual(self.decisions()[0]["device"], "Radeon™ RX 7900 — 显卡")

    def test_a_record_file_that_cannot_be_written_never_stops_the_tool(self):
        self.start_recording()
        self.record_file.mkdir()
        self.assertFalse(Benchmark_Record.record("alignment_plan"))
        ranked = Hardware_Utils.rank_benchmark_results(
            self.results(), higher_is_better=True, decision={"kind": "alignment_plan"})
        self.assertEqual(len(ranked), 3)

    def test_results_that_cannot_be_described_are_skipped_not_raised(self):
        self.start_recording()
        self.assertFalse(Benchmark_Record.record_ranking(
            [object()], [], higher_is_better=True, tie_fraction=0.03))
        self.assertFalse(Benchmark_Record.record_search_plans([object()], [], {}))
        self.assertFalse(Benchmark_Record.record_host_cache(object(), "auto"))
        self.assertFalse(self.record_file.exists())

    def test_a_line_a_killed_write_cut_short_is_skipped_when_reading(self):
        self.start_recording()
        Benchmark_Record.record("embedding_device", value=1)
        with open(self.record_file, "a", encoding="utf-8") as handle:
            handle.write('{"kind": "layout_dev')  # A process killed mid-write.
        Benchmark_Record.record("alignment_plan", value=2)  # Lands on the cut line, which stays unreadable.
        Benchmark_Record.record("injection_plan", value=3)
        self.assertEqual([decision["kind"] for decision in self.decisions()], ["embedding_device", "injection_plan"])
        self.assertEqual(Benchmark_Record.read_records(self.record_file.with_name("missing.jsonl")), [])

    def test_search_plans_record_predicted_seconds_and_unmeasured_plans(self):
        self.start_recording()
        timing = SimpleNamespace(setup=0.5, shutdown=0.25)
        serial = SearchPlan(cpu(), "serial", "float32", observations=[(timing, 0.01)])
        scalar = SearchPlan(gpu(), "scalar", "float32", 2, observations=[(timing, 0.001)])
        tiled = SearchPlan(gpu(), "tiled", "tf32", 4)
        costs = {0: 100, 1: 300}
        self.assertTrue(Benchmark_Record.record_search_plans(
            [serial, scalar, tiled], [scalar, serial], costs, notes=["tiled: unmeasured"], targets=2))
        [decision] = self.decisions()
        self.assertEqual((decision["kind"], decision["unit"], decision["direction"]), ("search_plan", "s", "lower"))
        self.assertEqual([candidate["value"] for candidate in decision["candidates"]], [4.75, 1.15, None])
        self.assertEqual([candidate["error"] for candidate in decision["candidates"]], [None, None, "not measured"])
        self.assertEqual(decision["candidates"][1]["precision"], "float32")
        self.assertEqual((decision["ranking"], decision["winner"]), ([1, 0], 1))
        self.assertEqual((decision["notes"], decision["targets"]), (["tiled: unmeasured"], 2))

    def test_the_host_cache_records_packed_or_tiled_embeddings(self):
        self.start_recording()
        for fully_cached, choice in ((True, "packed"), (False, "tiles")):
            store = SimpleNamespace(fully_cached=fully_cached, host_cache_bytes=4 << 30, total_source_bytes=123)
            Benchmark_Record.record_host_cache(store, "auto")
        self.assertEqual(
            [(decision["kind"], decision["choice"], decision["setting"], decision["limit_bytes"],
              decision["embedding_bytes"]) for decision in self.decisions()],
            [("host_cache", "packed", "auto", 4 << 30, 123), ("host_cache", "tiles", "auto", 4 << 30, 123)],
        )


class RecordedDecisionTests(RecordingFixture, unittest.TestCase):
    """The tools say what each Auto decision chooses, and the real code paths record it."""

    def test_every_caller_of_rank_benchmark_results_names_its_decision(self):
        kinds = {}
        for path in sorted(SRC_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name != "rank_benchmark_results":
                    continue
                relative = path.relative_to(SRC_DIR).as_posix()
                with self.subTest(caller=relative, line=node.lineno):
                    [decision] = [keyword.value for keyword in node.keywords if keyword.arg == "decision"]
                    self.assertIsInstance(decision, ast.Dict)
                    fields = {key.value: value for key, value in zip(decision.keys, decision.values)}
                    self.assertIn("unit", fields)
                    kinds[relative] = fields["kind"].value
        self.assertEqual(kinds, {
            "tools/Generate_Embeddings.py": "embedding_device",
            "tools/Align_Similarity_Matrix.py": "alignment_plan",
            "tools/Network_Injection.py": "injection_plan",
            "tools/Embedding_MSA.py": "msa_device",
            "utilities/Hardware_Acceleration.py": "layout_device",
        })

    def test_both_alignment_tools_record_their_host_cache_choice(self):
        # The choice is recorded after the store is made and before the job uses it.
        for name, store, used in (
            ("Align_Similarity_Matrix.py", "active_embedding_store",
             "active_matmul_precision = _resolve_active_matmul_precision("),
            ("Network_Injection.py", "embedding_store", "sequence_lengths = [shape[0] for shape in embedding_store"),
        ):
            with self.subTest(tool=name):
                source = (SRC_DIR / "tools" / name).read_text(encoding="utf-8")
                created = source.index(f"{store} = EmbeddingTileStore(")
                recorded = source.index(f"Benchmark_Record.record_host_cache({store}, HOST_CACHE_GB)")
                self.assertLess(created, recorded)
                self.assertLess(recorded, source.index(used, created))

    def precision_trial(self, devices, clock):
        """Run the alignment's automatic FP32/TF32 choice on a simulated device."""
        tasks = [(0, 1, "a", "b"), (0, 2, "a", "c")]
        results = [(0, 1, 1.0, 2, 2.0, 2), (0, 2, 1.5, 2, 2.5, 2)]
        safe = mock.Mock(feasible=True, projected_peak_bytes=10 << 30, safe_peak_bytes=13 << 30, tile_bytes=0,
                         transient_bytes=2 << 30, per_microbatch_bytes=512 << 20, reason="within reserve")
        clock = iter(clock)
        with mock.patch.object(similarity_matrix, "is_nvidia_cuda", side_effect=lambda device: device.type == "cuda"), \
                mock.patch.object(similarity_matrix, "EXECUTION_MODE", "scalar"), \
                mock.patch.object(similarity_matrix.Hardware_Utils, "get_available_devices", return_value=devices), \
                mock.patch.object(similarity_matrix.Hardware_Utils, "release_device_cache"), \
                mock.patch.object(similarity_matrix, "cuda_memory_plan",
                                  return_value=mock.Mock(free_bytes=12 << 30, total_bytes=16 << 30)), \
                mock.patch.object(similarity_matrix, "estimate_cuda_working_set", return_value=safe), \
                mock.patch.object(similarity_matrix, "_run_accelerated_pipeline", return_value=results), \
                mock.patch.object(similarity_matrix.time, "perf_counter", side_effect=lambda: next(clock)), \
                redirect_stdout(io.StringIO()):
            return similarity_matrix._resolve_active_matmul_precision(
                "auto", None, tasks, workers=2, store=mock.Mock(path="unused.h5"), sequence_lengths=[2, 2, 2])

    def test_the_alignment_records_its_fp32_tf32_trial_and_choice(self):
        self.start_recording()
        # FP32 takes 1.0 s for the two pairs and TF32 0.5 s: twice as fast.
        self.assertEqual(self.precision_trial([cpu(), gpu()], [0.0, 1.0, 1.0, 1.5]), "tf32")
        [decision] = self.decisions()
        self.assertEqual(
            {key: decision[key] for key in ("kind", "setting", "choice", "reason", "device", "spec", "pairs", "unit",
                                            "equivalent", "speedup", "required_speedup")},
            {"kind": "matmul_precision", "setting": "automatic_32bit", "choice": "tf32",
             "reason": "faster_and_equivalent", "device": "Test GPU (CUDA)", "spec": "cuda:0", "pairs": 2,
             "unit": "pairs/s", "equivalent": {"scalar": True}, "speedup": 2.0, "required_speedup": 1.1},
        )
        self.assertEqual(decision["rates"], [{"variant": "scalar", "precision": "float32", "value": 2.0},
                                             {"variant": "scalar", "precision": "tf32", "value": 4.0}])

    def test_a_tf32_trial_without_the_required_speedup_records_fp32(self):
        self.start_recording()
        self.assertEqual(self.precision_trial([gpu()], [0.0, 1.0, 1.0, 1.95]), "ieee_fp32")
        [decision] = self.decisions()
        self.assertEqual((decision["choice"], decision["reason"]), ("ieee_fp32", "too_little_speedup"))

    def test_a_machine_without_nvidia_cuda_records_fp32_and_why(self):
        self.start_recording()
        self.assertEqual(self.precision_trial([cpu()], []), "ieee_fp32")
        [decision] = self.decisions()
        self.assertEqual((decision["kind"], decision["choice"], decision["reason"]),
                         ("matmul_precision", "ieee_fp32", "no_nvidia_cuda"))

    def test_the_database_search_records_its_final_plan_ranking(self):
        self.start_recording(stage="7")
        tasks = [(index, f"h{index}", f"h{index}", np.ones((3, 4), np.float32), "local", -2.0, "longer_sequence",
                  "ACD", "ACD") for index in range(40)]

        def execute(plan, selected, *args, timing=None):
            if timing is not None:
                timing.setup, timing.processing, timing.shutdown = 0.001, 0.0001 * len(selected), 0.001
            return [{"index": int(task[0]), "raw_score": 1.0, "norm_score": 1.0, "aln_len": 1} for task in selected]

        with mock.patch.object(ssearch, "DEVICE_SELECTION", "auto"), \
                mock.patch.object(ssearch, "ACCELERATOR_PRECISION", "automatic_32bit"), \
                mock.patch.object(ssearch.Hardware_Utils, "get_available_devices", return_value=[cpu()]), \
                mock.patch.object(ssearch.Hardware_Utils, "release_device_cache"), \
                mock.patch.object(ssearch, "_execute_search_plan", side_effect=execute), \
                redirect_stdout(io.StringIO()):
            ranked = ssearch._select_search_plans(
                tasks, 1, "unused.h5", mock.Mock(feature_dimension=4, dtypes=[np.dtype("float32")]),
                [3] * len(tasks), tasks[0][3])
        [decision] = self.decisions()
        self.assertEqual((decision["kind"], decision["stage"], decision["targets"], decision["query_length"]),
                         ("search_plan", "7", 40, 3))
        self.assertEqual([(candidate["spec"], candidate["variant"]) for candidate in decision["candidates"]],
                         [("cpu", "serial")])
        self.assertEqual(len(decision["ranking"]), len(ranked))
        self.assertEqual(decision["winner"], 0)
        self.assertGreater(decision["candidates"][0]["value"], 0)
        json.dumps(decision)


if __name__ == "__main__":
    unittest.main()
