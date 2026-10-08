# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Hardware_Acceleration's runtime device policy (enumeration, installer
filtering, selection, benchmark ranking) and how the embedding, alignment and
layout workloads use it."""

from contextlib import redirect_stdout
import io
import json
import pathlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import h5py
import numpy as np
import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
UTILITIES = SRC / "utilities"
TOOLS = SRC / "tools"
for path in (str(SRC), str(UTILITIES), str(TOOLS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import Generate_Embeddings
from utilities import Hardware_Acceleration as Hardware_Utils
import Align_Similarity_Matrix as Alignment


class HardwareUtilityTests(unittest.TestCase):
    def test_enumerates_every_visible_backend_with_stable_specs(self):
        with mock.patch.object(Hardware_Utils, "_validated_device_specs", return_value=None), \
                mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "device_count", return_value=2), \
                mock.patch.object(
                    torch.cuda,
                    "get_device_name",
                    side_effect=lambda index: f"CUDA {index}",
                ), mock.patch.object(torch.xpu, "is_available", return_value=True), \
                mock.patch.object(torch.xpu, "device_count", return_value=1), \
                mock.patch.object(
                    torch.xpu, "get_device_name", return_value="Intel Arc"
                ), mock.patch.object(
                    torch.backends.mps, "is_available", return_value=True
                ), mock.patch.object(
                    torch.backends.mps,
                    "get_name",
                    return_value="Apple GPU",
                    create=True,
                ):
            candidates = Hardware_Utils.get_available_devices()

        self.assertEqual(
            [candidate.spec for candidate in candidates],
            [
                "cpu",
                "cuda:0",
                "cuda:1",
                "xpu:0",
                "mps",
            ],
        )
        self.assertTrue(candidates[1].supports_streams)
        self.assertTrue(candidates[3].supports_streams)
        self.assertFalse(candidates[-1].supports_streams)

    def test_integrated_arc_benchmark_can_choose_cpu_or_xpu(self):
        cpu = Hardware_Utils.DeviceCandidate("cpu", "CPU", torch.device("cpu"), "cpu")
        arc = Hardware_Utils.DeviceCandidate(
            "xpu:0", "Intel Arc 140V", torch.device("xpu:0"), "xpu", 0, True
        )
        for arc_seconds, expected in ((2.0, "cpu"), (0.5, "xpu:0"), (0.99, "cpu")):
            with self.subTest(arc_seconds=arc_seconds):
                ranked = Hardware_Utils.rank_benchmark_results(
                    [Hardware_Utils.BenchmarkResult(arc, arc_seconds),
                     Hardware_Utils.BenchmarkResult(cpu, 1.0)],
                    higher_is_better=False,
                )
                self.assertEqual(ranked[0].candidate.spec, expected)

    def test_legacy_directml_selection_migrates_to_auto(self):
        self.assertEqual(Hardware_Utils.normalize_device_selection("directml"), "auto")
        self.assertEqual(Hardware_Utils.normalize_device_selection("directml:2"), "auto")
        self.assertEqual(
            Hardware_Utils.normalize_device_selection("Old GPU [directml:0]"),
            "auto",
        )

    def test_selections_and_display_labels_normalize_to_persisted_specs(self):
        cases = {
            None: "auto",
            "": "auto",
            "Auto Benchmark": "auto",
            "AUTO": "auto",
            "CUDA": "cuda:0",
            "xpu": "xpu:0",
            "cuda:3": "cuda:3",
            "MPS": "mps",
            "RTX 4090 (CUDA) [cuda:1]": "cuda:1",
            "CPU [cpu]": "cpu",
        }
        for selection, expected in cases.items():
            with self.subTest(selection=selection):
                self.assertEqual(
                    Hardware_Utils.normalize_device_selection(selection), expected
                )

    def test_tie_margin_prefers_cpu_then_fewer_lanes(self):
        cpu = Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        gpu = Hardware_Utils.DeviceCandidate(
            "cuda:0", "GPU", torch.device("cuda:0"), "cuda", 0, True
        )
        ranked = Hardware_Utils.rank_benchmark_results(
            [
                Hardware_Utils.BenchmarkResult(gpu, 100.0, lanes=8),
                Hardware_Utils.BenchmarkResult(gpu, 99.0, lanes=2),
                Hardware_Utils.BenchmarkResult(cpu, 97.1, lanes=1),
            ],
            higher_is_better=True,
        )
        self.assertEqual(ranked[0].candidate.spec, "cpu")

    def test_tie_margin_prefers_scalar_variant_when_tiled_gain_is_noise(self):
        gpu = Hardware_Utils.DeviceCandidate(
            "cuda:0", "GPU", torch.device("cuda:0"), "cuda", 0, True
        )
        ranked = Hardware_Utils.rank_benchmark_results(
            [
                Hardware_Utils.BenchmarkResult(
                    gpu, 100.0, lanes=2, variant="tiled"
                ),
                Hardware_Utils.BenchmarkResult(
                    gpu, 98.0, lanes=2, variant="scalar"
                ),
            ],
            higher_is_better=True,
        )
        self.assertEqual(ranked[0].variant, "scalar")
        self.assertEqual(ranked[1].lanes, 2)

    def test_tiled_tie_prefers_peak_memory_then_lanes_then_tile_memory(self):
        gpu = Hardware_Utils.DeviceCandidate(
            "cuda:0", "GPU", torch.device("cuda:0"), "cuda", 0, True
        )
        plans = [
            types.SimpleNamespace(microbatch_workspace_bytes=900),
            types.SimpleNamespace(microbatch_workspace_bytes=800),
            types.SimpleNamespace(microbatch_workspace_bytes=700),
            types.SimpleNamespace(microbatch_workspace_bytes=600),
        ]
        ranked = Hardware_Utils.rank_benchmark_results(
            [
                Hardware_Utils.BenchmarkResult(
                    gpu,
                    100.0,
                    lanes=4,
                    variant="tiled",
                    execution_plan=plans[0],
                    peak_memory_bytes=1000,
                ),
                Hardware_Utils.BenchmarkResult(
                    gpu,
                    98.0,
                    lanes=2,
                    variant="tiled",
                    execution_plan=plans[1],
                    peak_memory_bytes=900,
                ),
                Hardware_Utils.BenchmarkResult(
                    gpu,
                    97.5,
                    lanes=1,
                    variant="tiled",
                    execution_plan=plans[2],
                    peak_memory_bytes=900,
                ),
                Hardware_Utils.BenchmarkResult(
                    gpu,
                    97.3,
                    lanes=1,
                    variant="tiled",
                    execution_plan=plans[3],
                    peak_memory_bytes=900,
                ),
            ],
            higher_is_better=True,
        )
        self.assertEqual(ranked[0].value, 97.3)

    def test_manual_unavailable_device_is_not_silently_replaced(self):
        with self.assertRaisesRegex(ValueError, "not available"):
            Hardware_Utils.resolve_device_selection(
                "xpu:7",
                [
                    Hardware_Utils.DeviceCandidate(
                        "cpu", "CPU", torch.device("cpu"), "cpu"
                    )
                ],
            )

    def test_synchronization_routes_by_backend(self):
        cuda = Hardware_Utils.DeviceCandidate(
            "cuda:0", "GPU", torch.device("cuda:0"), "cuda", 0, True
        )
        xpu = Hardware_Utils.DeviceCandidate(
            "xpu:0", "Arc", torch.device("xpu:0"), "xpu", 0, True
        )
        with mock.patch.object(torch.cuda, "synchronize") as cuda_sync, \
                mock.patch.object(torch.xpu, "synchronize") as xpu_sync:
            Hardware_Utils.synchronize_device(cuda)
            Hardware_Utils.synchronize_device(xpu)
        cuda_sync.assert_called_once_with(cuda.device)
        xpu_sync.assert_called_once_with(xpu.device)


class EmbeddingHardwareTests(unittest.TestCase):
    @staticmethod
    def _write_fasta(path):
        path.write_text(
            ">short\nACD\n>medium\nACDEFG\n>long\nACDEFGHIK\n",
            encoding="utf-8",
        )

    def test_percentile_samples_are_real_pending_sequences(self):
        pending = [(str(index), "A" * length) for index, length in enumerate(
            [2, 3, 5, 8, 13, 21, 34, 55]
        )]
        samples = Generate_Embeddings._representative_sequences(pending)
        self.assertEqual([len(sample) for sample in samples], [5, 13, 34])
        self.assertTrue(all(sample in dict(pending).values() for sample in samples))

    def test_remote_model_skips_resolution_and_benchmark_requests(self):
        class RemotePlugin:
            SUPPORTED_MODELS = ["remote"]
            MODEL_EXECUTION_MODES = {"remote": "remote_api"}

            def __init__(self):
                self.loads = 0
                self.requests = []

            def load_model(self, model_name, device):
                self.loads += 1
                self.asserted_device = device
                return object()

            def get_embedding(self, sequence, model, device, dtype):
                self.requests.append(sequence)
                return np.ones((len(sequence), 4), dtype=dtype)

        with tempfile.TemporaryDirectory() as temp_dir:
            fasta = pathlib.Path(temp_dir) / "input.fasta"
            output = pathlib.Path(temp_dir) / "output.h5"
            self._write_fasta(fasta)
            plugin = RemotePlugin()
            with mock.patch.object(
                Generate_Embeddings, "DEVICE_SELECTION", "cuda:99"
            ), mock.patch.object(
                Generate_Embeddings.Hardware_Utils,
                "get_available_devices",
                side_effect=AssertionError("remote device enumeration ran"),
            ), mock.patch.object(
                Generate_Embeddings.Hardware_Utils,
                "resolve_device_selection",
                side_effect=AssertionError("remote device resolution ran"),
            ), mock.patch.object(
                Generate_Embeddings,
                "_benchmark_embedding_devices",
                side_effect=AssertionError("remote benchmark ran"),
            ):
                count = Generate_Embeddings.generate_embeddings(
                    fasta,
                    output,
                    "remote",
                    "float32",
                    plugin_loader=lambda _: plugin,
                )

        self.assertEqual(count, 3)
        self.assertEqual(plugin.loads, 1)
        self.assertIsNone(plugin.asserted_device)
        self.assertEqual(plugin.requests, ["ACD", "ACDEFG", "ACDEFGHIK"])

    def test_auto_production_failure_retries_next_ranked_device(self):
        class LocalPlugin:
            SUPPORTED_MODELS = ["local"]
            MODEL_EXECUTION_MODES = {"local": "local"}

            def load_model(self, model_name, device):
                return object()

            def get_embedding(self, sequence, model, device, dtype):
                if str(device).startswith("cuda"):
                    raise RuntimeError("simulated accelerator failure")
                return np.ones((len(sequence), 2), dtype=dtype)

        cpu = Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        cuda = Hardware_Utils.DeviceCandidate(
            "cuda:0", "GPU", torch.device("cuda:0"), "cuda", 0, True
        )
        ranked = [
            Hardware_Utils.BenchmarkResult(cuda, 1.0),
            Hardware_Utils.BenchmarkResult(cpu, 2.0),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta = pathlib.Path(temp_dir) / "input.fasta"
            output = pathlib.Path(temp_dir) / "output.h5"
            fasta.write_text(">only\nACDE\n", encoding="utf-8")
            with mock.patch.object(
                Generate_Embeddings.Hardware_Utils,
                "get_available_devices",
                return_value=[cpu, cuda],
            ), mock.patch.object(
                Generate_Embeddings,
                "_benchmark_embedding_devices",
                return_value=(object(), ranked),
            ), mock.patch.object(
                Generate_Embeddings, "DEVICE_SELECTION", "auto"
            ):
                generated = Generate_Embeddings.generate_embeddings(
                    fasta,
                    output,
                    "local",
                    "float32",
                    plugin_loader=lambda _: LocalPlugin(),
                )
            with h5py.File(output, "r") as handle:
                self.assertIn("only", handle["embeddings"])
        self.assertEqual(generated, 1)


class AlignmentHardwareTests(unittest.TestCase):
    def test_manual_cpu_bypasses_sampling_and_lane_tuning(self):
        cpu = Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        with mock.patch.object(Alignment, "DEVICE_SELECTION", "cpu"), \
                mock.patch.object(Alignment, "EXECUTION_MODE", "auto"), \
                mock.patch.object(
                    Alignment.Hardware_Utils,
                    "get_available_devices",
                    return_value=[cpu],
                ), mock.patch.object(
                    Alignment,
                    "_select_accelerator_lanes",
                    side_effect=AssertionError("manual CPU tuning ran"),
                ):
            plans = Alignment._benchmark_processing_plans(
                [(0, 1, "a", "b")], 4, "unused.h5", 0
            )
        self.assertEqual(plans[0].candidate.spec, "cpu")

    def test_backend_lane_candidates_match_supported_stream_controls(self):
        self.assertEqual(
            Alignment._accelerator_lane_candidates(torch.device("cuda:0"), 32),
            [1, 2, 4, 8, 16],
        )
        self.assertEqual(
            Alignment._accelerator_lane_candidates(torch.device("xpu:0"), 32),
            [1, 2, 4],
        )
        self.assertEqual(
            Alignment._accelerator_lane_candidates(torch.device("mps"), 32),
            [1],
        )
        unknown_device = types.SimpleNamespace(type="privateuseone")
        self.assertEqual(
            Alignment._accelerator_lane_candidates(unknown_device, 32), [1]
        )

    def test_accelerator_lanes_are_always_benchmarked_automatically(self):
        device = types.SimpleNamespace(type="cuda")
        tasks = [(0, 1), (0, 2), (1, 2), (0, 3)]

        def complete_benchmark(*_args, **kwargs):
            timer = kwargs["benchmark_trial"]
            timer.completed = len(_args[0])
            timer.started_at = 0.0
            timer.stopped_at = 1.0

        # Start from an empty lane cache and leave no "Test GPU" entry behind.
        with mock.patch.dict(Alignment.accelerator_lane_cache, clear=True), mock.patch.object(
            Alignment,
            "_accelerator_lane_candidates",
            return_value=[1, 2],
        ), mock.patch.object(
            Alignment,
            "_accelerator_name",
            return_value="Test GPU",
        ), mock.patch.object(
            Alignment,
            "_run_accelerated_pipeline",
            side_effect=complete_benchmark,
        ) as run_pipeline:
            selected = Alignment._select_accelerator_lanes(
                tasks, 4, "input.h5", device, 0
            )

        self.assertEqual(selected, 1)
        self.assertEqual(
            [call.kwargs["accelerator_workers"] for call in run_pipeline.call_args_list],
            [1, 2],
        )
        self.assertEqual(
            [call.args[0] for call in run_pipeline.call_args_list],
            [tasks, tasks],
        )
        self.assertFalse(hasattr(Alignment, "ACCELERATOR_LANES"))

class LayoutHardwareTests(unittest.TestCase):
    def test_size_class_boundaries(self):
        self.assertEqual(Hardware_Utils.layout_size_class(499), "small")
        self.assertEqual(Hardware_Utils.layout_size_class(500), "medium")
        self.assertEqual(Hardware_Utils.layout_size_class(2000), "medium")
        self.assertEqual(Hardware_Utils.layout_size_class(2001), "massive")
        self.assertEqual(Hardware_Utils.benchmark_step_count("small"), 20)
        self.assertEqual(Hardware_Utils.benchmark_step_count("medium"), 10)
        self.assertEqual(Hardware_Utils.benchmark_step_count("massive"), 3)
        self.assertEqual(Hardware_Utils.estimated_stage_steps({"MAX_STEPS": 10000}), 2500)
        self.assertEqual(Hardware_Utils.estimated_stage_steps({"MAX_STEPS": 2}), 1)

    def _rank_with_fake_clock(self, max_steps):
        """Benchmark a CPU (cheap setup, slow steps) against a GPU (slow setup,
        fast steps) on a fake clock; return the specs best first and the log."""
        clock = [0.0]
        costs = {"cpu": (0.001, 0.060), "cuda:0": (0.5, 0.002)}

        def simulation_class(spec):
            class FakeSimulation:
                def __init__(self, *args, **kwargs):
                    clock[0] += costs[spec][0]

                def step(self, step):
                    clock[0] += costs[spec][1]

                def get_pos(self):
                    return None
            return FakeSimulation

        cpu = Hardware_Utils.DeviceCandidate("cpu", "CPU", torch.device("cpu"), "cpu")
        gpu = Hardware_Utils.DeviceCandidate("cuda:0", "GPU", torch.device("cpu"), "cuda", 0, True)
        prepared = Hardware_Utils.PreparedLayoutBatch(
            global_nodes=np.arange(3), edges=np.array([[0, 1], [1, 2]]),
            scores=np.ones(2), positions=np.zeros((3, 2), np.float32),
            component_labels=np.zeros(3, np.int32), box_limits=np.full(3, 10.0, np.float32),
            node_count=3, is_large_job=False)
        output = io.StringIO()
        with mock.patch.object(Hardware_Utils, "get_available_devices", return_value=[cpu, gpu]), \
                mock.patch.object(Hardware_Utils, "synchronize_device"), \
                mock.patch.object(Hardware_Utils, "release_device_cache"), \
                mock.patch.object(Hardware_Utils, "_snapshot_random_state"), \
                mock.patch.object(Hardware_Utils, "_restore_random_state"), \
                mock.patch.object(Hardware_Utils.time, "perf_counter", side_effect=lambda: clock[0]), \
                redirect_stdout(output):
            ranked = Hardware_Utils.benchmark_layout_devices(
                prepared, {"MAX_STEPS": max_steps, "SIMILARITY_THRESHOLD": 0.0},
                selection="auto", size_class="massive", engine_label="SSN",
                cpu_simulation_class=simulation_class("cpu"),
                gpu_simulation_class=simulation_class("cuda:0"))
        return [result.candidate.spec for result in ranked], ranked, output.getvalue()

    def test_benchmark_ranks_devices_by_setup_plus_a_stage_of_steps(self):
        # 2500 steps: CPU 0.001 + 2500 x 0.060 = 150 s, GPU 0.5 + 2500 x 0.002 = 5.5 s.
        order, ranked, log = self._rank_with_fake_clock(10000)
        self.assertEqual(order, ["cuda:0", "cpu"])
        self.assertAlmostEqual(ranked[0].value, 5.5, places=6)
        self.assertIn("stage = setup + 2500 steps", log)
        # A two-step stage: CPU 0.121 s beats the GPU's 0.504 s setup.
        order, ranked, _ = self._rank_with_fake_clock(8)
        self.assertEqual(order, ["cpu", "cuda:0"])
        self.assertAlmostEqual(ranked[0].value, 0.121, places=6)

    def test_representative_is_median_cost_in_each_populated_class(self):
        sizes = [100, 300, 500, 1000, 2000, 2001, 3000]
        components = []
        node_to_component = {}
        component_edges = {}
        next_node = 0
        for component_index, size in enumerate(sizes):
            component = list(range(next_node, next_node + size))
            next_node += size
            components.append(component)
            for node in component:
                node_to_component[node] = component_index
            component_edges[component_index] = [(component[0], component[1])]
        jobs = [[component] for component in components]
        selected = Hardware_Utils.representative_job_indices(
            jobs,
            node_to_component,
            component_edges,
        )
        self.assertEqual(selected, {"small": 1, "medium": 3, "massive": 6})

    def test_representative_preparation_preserves_numpy_random_state(self):
        jobs = [[[0, 1]]]
        before = np.random.get_state()
        # Hide the accelerators: snapshotting their RNG state would otherwise
        # create a CUDA/XPU context on the developer's GPU.
        with mock.patch.object(torch.cuda, "is_available", return_value=False), \
                mock.patch.object(torch.xpu, "is_available", return_value=False), \
                mock.patch.object(torch.backends.mps, "is_available", return_value=False):
            # Use keyword order explicitly to make the contract easy to audit.
            prepared = Hardware_Utils.prepare_representative_batches(
                jobs=jobs,
                representative_indices={"small": 0},
                node_to_component={0: 0, 1: 0},
                component_edges={0: [(0, 1)]},
                component_scores={0: [1.0]},
                params={"BOX_SCALE": 1.0},
            )
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(prepared["small"].positions.shape, (2, 2))

    def test_gpu_constructors_accept_an_explicit_device(self):
        import inspect
        import Layout_Engine_SSN as ssn_layout

        if ssn_layout.HAS_TORCH:
            self.assertIn(
                "device", inspect.signature(ssn_layout.SSNSimulationGPU).parameters
            )

    def test_ssn_layout_runs_manual_cpu_without_benchmarking(self):
        import Layout_Engine_SSN as ssn_layout

        connectivity = np.array(
            [[0, 1, 1.0], [1, 2, 1.0]], dtype=np.float32
        )
        common = {
            "LAYOUT_DEVICE_SELECTION": "cpu",
            "SIMILARITY_THRESHOLD": 0.0,
            "MAX_STEPS": 1,
            "RMSD_WINDOW": 2,
            "BOX_SCALE": 1.0,
            "PACKING_GRID_SIZE": 20.0,
            "PACKING_PADDING": 5.0,
            "PACKING_GEOMETRY": "Square",
        }
        cpu = Hardware_Utils.DeviceCandidate("cpu", "CPU", torch.device("cpu"), "cpu")
        # The real enumeration would read the developer's installer state and
        # initialise CUDA; the manual CPU selection needs only the CPU entry.
        with mock.patch.object(
            ssn_layout.Layout_Hardware, "get_available_devices", return_value=[cpu]
        ), mock.patch.object(
            ssn_layout.Layout_Hardware,
            "benchmark_layout_devices",
            side_effect=AssertionError("manual layout benchmark ran"),
        ):
            layout_positions, _ = ssn_layout.calculate_layout(
                connectivity, 3, common.copy()
            )
        self.assertEqual(layout_positions.shape, (3, 2))

    def test_auto_layout_stage_restarts_on_next_ranked_plan(self):
        import Layout_Engine_SSN as ssn_layout

        cpu = Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        cuda = Hardware_Utils.DeviceCandidate(
            "cuda:0", "GPU", torch.device("cuda:0"), "cuda", 0, True
        )
        rankings = {
            "small": [
                Hardware_Utils.BenchmarkResult(cuda, 1.0),
                Hardware_Utils.BenchmarkResult(cpu, 2.0),
            ]
        }
        connectivity = np.array(
            [[0, 1, 1.0], [1, 2, 1.0]], dtype=np.float32
        )

        def simulated_stage(candidate, positions, *args, **kwargs):
            if candidate.backend == "cuda":
                raise RuntimeError("simulated device loss")
            return positions.copy()

        with mock.patch.object(
            ssn_layout.Layout_Hardware,
            "manual_layout_rankings",
            return_value=rankings,
        ), mock.patch.object(
            ssn_layout, "_run_layout_stage", side_effect=simulated_stage
        ) as run_stage:
            positions, _ = ssn_layout.calculate_layout(
                connectivity,
                3,
                {
                    "LAYOUT_DEVICE_SELECTION": "auto",
                    "SIMILARITY_THRESHOLD": 0.0,
                    "PACKING_GRID_SIZE": 20.0,
                    "PACKING_PADDING": 5.0,
                    "PACKING_GEOMETRY": "Square",
                },
            )
        self.assertEqual(run_stage.call_count, 2)
        self.assertEqual(positions.shape, (3, 2))


class GuiContractTests(unittest.TestCase):
    def test_remote_and_layout_device_controls_are_visible_and_persist_specs(self):
        tools_source = (SRC / "EMAPSSN_Tools.py").read_text(encoding="utf-8")
        config_source = (SRC / "EMAPSSN_Config.py").read_text(encoding="utf-8")
        generator_source = (SRC / "Layout_Cache_Generator.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Remote API — local device not applicable", tools_source)
        self.assertNotIn("ACCELERATOR_LANES", tools_source)
        self.assertIn('LAYOUT_DEVICE_SELECTION = "auto"', config_source)
        self.assertIn("widget.currentData()", config_source)
        self.assertIn('"LAYOUT_DEVICE_SELECTION"', generator_source)


class RuntimeFilteringTests(unittest.TestCase):
    def test_validated_state_is_read_from_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "ssn_backend.json").write_text(
                json.dumps({"schema": 3, "validated_devices": [{"spec": "cuda:1", "success": True}]}),
                encoding="utf-8",
            )
            with mock.patch.object(Hardware_Utils.sys, "prefix", folder):
                self.assertEqual(Hardware_Utils._validated_device_specs(), {"cuda:1"})

    def test_unvalidated_visible_gpu_is_filtered(self):
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_name=lambda index: f"GPU {index}",
        )
        fake_mps = types.SimpleNamespace(is_available=lambda: False)
        fake_torch = types.SimpleNamespace(
            cuda=fake_cuda,
            version=types.SimpleNamespace(hip=None),
            backends=types.SimpleNamespace(mps=fake_mps),
            device=lambda value: value,
        )
        with mock.patch.object(Hardware_Utils, "torch", fake_torch), mock.patch.object(
            Hardware_Utils, "_validated_device_specs", return_value={"cuda:1"}
        ):
            candidates = Hardware_Utils.get_available_devices()
        self.assertEqual([candidate.spec for candidate in candidates], ["cpu", "cuda:1"])

    def _approved_specs(self, state):
        """Read an installer state file (None: no file) from a temporary prefix."""
        with tempfile.TemporaryDirectory() as folder:
            if state is not None:
                Path(folder, "ssn_backend.json").write_text(
                    json.dumps(state), encoding="utf-8"
                )
            with mock.patch.object(Hardware_Utils.sys, "prefix", folder):
                return Hardware_Utils._validated_device_specs()

    def test_installer_state_decides_which_devices_are_approved(self):
        passed = {"spec": "cuda:0", "success": True}
        failed = {"spec": "cuda:1", "success": False}
        # None means unmanaged: every visible device is offered.
        self.assertIsNone(self._approved_specs(None))
        self.assertIsNone(self._approved_specs({"schema": 2, "validated_devices": [passed]}))
        self.assertIsNone(self._approved_specs({"schema": 6}))
        # A managed environment that validated no device approves none.
        self.assertEqual(self._approved_specs({"schema": 6, "validated_devices": []}), set())
        self.assertEqual(
            self._approved_specs(
                {"schema": 6, "validated_devices": [passed, failed, {"success": True}]}
            ),
            {"cuda:0"},
        )

    def test_corrupt_schema_is_unvalidated_state(self):
        # A hand-edited "schema" leaves the environment unmanaged, like an
        # unreadable state file, instead of crashing every device enumeration.
        passed = [{"spec": "cuda:0", "success": True}]
        for schema in (None, [6], "abc", float("nan"), float("inf")):
            with self.subTest(schema=schema):
                self.assertIsNone(
                    self._approved_specs({"schema": schema, "validated_devices": passed})
                )
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 2,
                get_device_name=lambda index: f"GPU {index}",
            ),
            version=types.SimpleNamespace(hip=None),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
            device=lambda value: value,
        )
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "ssn_backend.json").write_text(
                json.dumps({"schema": None, "validated_devices": passed}), encoding="utf-8"
            )
            with mock.patch.object(Hardware_Utils.sys, "prefix", folder), \
                    mock.patch.object(Hardware_Utils, "torch", fake_torch):
                candidates = Hardware_Utils.get_available_devices()
        self.assertEqual([candidate.spec for candidate in candidates], ["cpu", "cuda:0", "cuda:1"])

    def test_optimal_device_prefers_cuda_then_xpu_then_mps_among_approved(self):
        def runtime(name):
            return types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_name=lambda index: f"{name} {index}",
            )

        fake_torch = types.SimpleNamespace(
            cuda=runtime("GPU"),
            xpu=runtime("Arc"),
            version=types.SimpleNamespace(hip=None),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: True, get_name=lambda: "Apple")
            ),
            device=lambda value: value,
        )
        cases = (
            (None, ["cpu", "cuda:0", "xpu:0", "mps"], "cuda:0"),
            ({"xpu:0", "mps"}, ["cpu", "xpu:0", "mps"], "xpu:0"),
            ({"mps"}, ["cpu", "mps"], "mps"),
            (set(), ["cpu"], "cpu"),
        )
        for approved, specs, optimal in cases:
            with self.subTest(approved=approved), mock.patch.object(
                Hardware_Utils, "torch", fake_torch
            ), mock.patch.object(
                Hardware_Utils, "_validated_device_specs", return_value=approved
            ):
                self.assertEqual(
                    [candidate.spec for candidate in Hardware_Utils.get_available_devices()],
                    specs,
                )
                self.assertEqual(Hardware_Utils.get_optimal_device(), optimal)


if __name__ == "__main__":
    unittest.main()
