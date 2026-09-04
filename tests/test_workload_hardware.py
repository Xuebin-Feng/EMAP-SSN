import pathlib
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
import Hardware_Utils
import Layout_Hardware
import PLM_Plugin_Utils
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

    def test_legacy_directml_selection_migrates_to_auto(self):
        self.assertEqual(Hardware_Utils.normalize_device_selection("directml"), "auto")
        self.assertEqual(Hardware_Utils.normalize_device_selection("directml:2"), "auto")
        self.assertEqual(
            Hardware_Utils.normalize_device_selection("Old GPU [directml:0]"),
            "auto",
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


class PluginContractTests(unittest.TestCase):
    def test_every_installed_plugin_has_complete_static_declaration(self):
        modes = PLM_Plugin_Utils.discover_model_execution_modes(
            SRC / "resources" / "pLM_models"
        )
        self.assertEqual(modes["esmc_6b"], "remote_api")
        self.assertEqual(modes["esmc_300m"], "local")
        self.assertIn("esm2_t48_15b", modes)

    def test_missing_and_unknown_modes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = pathlib.Path(temp_dir) / "missing.py"
            missing.write_text(
                'SUPPORTED_MODELS = ["local_model"]\n'
                'MODEL_EXECUTION_MODES = {}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                PLM_Plugin_Utils.read_plugin_metadata(missing)

            unknown = pathlib.Path(temp_dir) / "unknown.py"
            unknown.write_text(
                'SUPPORTED_MODELS = ["future_model"]\n'
                'MODEL_EXECUTION_MODES = {"future_model": "cloud"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unknown"):
                PLM_Plugin_Utils.read_plugin_metadata(unknown)


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
        Alignment.accelerator_lane_cache.clear()
        device = types.SimpleNamespace(type="cuda")
        tasks = [(0, 1), (0, 2), (1, 2), (0, 3)]

        def complete_benchmark(*_args, **kwargs):
            timer = kwargs["benchmark_timer"]
            timer.started_at = 0.0
            timer.stopped_at = 1.0

        with mock.patch.object(
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
            [call.kwargs["warmup_task_count"] for call in run_pipeline.call_args_list],
            [2, 2],
        )
        self.assertFalse(hasattr(Alignment, "ACCELERATOR_LANES"))

class LayoutHardwareTests(unittest.TestCase):
    def test_size_class_boundaries(self):
        self.assertEqual(Layout_Hardware.layout_size_class(499), "small")
        self.assertEqual(Layout_Hardware.layout_size_class(500), "medium")
        self.assertEqual(Layout_Hardware.layout_size_class(2000), "medium")
        self.assertEqual(Layout_Hardware.layout_size_class(2001), "massive")
        self.assertEqual(Layout_Hardware.benchmark_step_count("small"), 20)
        self.assertEqual(Layout_Hardware.benchmark_step_count("medium"), 5)
        self.assertEqual(Layout_Hardware.benchmark_step_count("massive"), 1)

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
        selected = Layout_Hardware.representative_job_indices(
            jobs,
            node_to_component,
            component_edges,
            {},
            engine="molecular_dynamics",
        )
        self.assertEqual(selected, {"small": 1, "medium": 3, "massive": 6})

    def test_representative_preparation_preserves_numpy_random_state(self):
        jobs = [[[0, 1]]]
        before = np.random.get_state()
        # Use keyword order explicitly to make the contract easy to audit.
        prepared = Layout_Hardware.prepare_representative_batches(
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
        import Layout_Engine_SSN_MolecularDynamics as molecular
        import Layout_Engine_SSN_MonteCarlo as monte_carlo

        if molecular.HAS_TORCH:
            self.assertIn(
                "device", inspect.signature(molecular.SSNSimulationGPU).parameters
            )
        if monte_carlo.HAS_TORCH:
            self.assertIn(
                "device",
                inspect.signature(monte_carlo.SSNSimulationGPU).parameters,
            )

    def test_both_physics_engines_run_manual_cpu_without_benchmarking(self):
        import Layout_Engine_SSN_MolecularDynamics as molecular
        import Layout_Engine_SSN_MonteCarlo as monte_carlo

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
            "MC_SWEEPS": 1,
            "MC_QUENCH_SWEEPS": 0,
            "MC_RANDOM_SEED": 42,
        }
        with mock.patch.object(
            molecular.Layout_Hardware,
            "benchmark_layout_devices",
            side_effect=AssertionError("manual layout benchmark ran"),
        ), mock.patch.object(
            monte_carlo.Layout_Hardware,
            "benchmark_layout_devices",
            side_effect=AssertionError("manual layout benchmark ran"),
        ):
            molecular_positions, _ = molecular.calculate_layout(
                connectivity, 3, common.copy()
            )
            monte_positions, _ = monte_carlo.calculate_layout(
                connectivity, 3, common.copy()
            )
        self.assertEqual(molecular_positions.shape, (3, 2))
        self.assertEqual(monte_positions.shape, (3, 2))

    def test_auto_layout_stage_restarts_on_next_ranked_plan(self):
        import Layout_Engine_SSN_MolecularDynamics as molecular

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
            molecular.Layout_Hardware,
            "manual_layout_rankings",
            return_value=rankings,
        ), mock.patch.object(
            molecular, "_run_layout_stage", side_effect=simulated_stage
        ) as run_stage:
            positions, _ = molecular.calculate_layout(
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

    def test_align_similarity_matrix_has_batch_size_control(self):
        tools_source = (SRC / "EMAPSSN_Tools.py").read_text(encoding="utf-8")
        calculation_tools = tools_source.split(
            '"Sequence_Similarity_Calculations": {', 1
        )[1]
        align_controls = calculation_tools.split(
            '"Align_Similarity_Matrix.py": [', 1
        )[1].split('"Align_Substitution_Matrix.py": [', 1)[0]
        self.assertIn('"var_name": "BATCH_SIZE"', align_controls)


if __name__ == "__main__":
    unittest.main()
