"""Tests of Network_Injection: input/output paths and embedding metadata,
execution-variant filtering, first-batch benchmark trials, CPU and accelerator
batch production, resume scans, network compilation (including the mapping of
reused old edges onto a reordered sequence set) and main() exit codes.
"""
import io
import os
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
UTILITIES_DIR = os.path.join(PROJECT_ROOT, "src", "utilities")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "src", "tools")
for path in (SRC_DIR, UTILITIES_DIR, TOOLS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# The tests package points tool imports at a missing settings file.
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    from utilities import HDF5_Storage as Embedding_HDF5
    from utilities import Prefilter_Calibration
    import Network_Injection as network_injection

from tests.alignment_fixtures import BatchResumeScanFixture, RecordingTrial  # noqa: E402


class NetworkInjectionPipelineTests(unittest.TestCase):
    def test_execution_mode_filters_injection_candidate_variants(self):
        cpu = network_injection.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        cuda = network_injection.Hardware_Utils.DeviceCandidate(
            "cuda:0", "CUDA", torch.device("cuda:0"), "cuda"
        )
        xpu = network_injection.Hardware_Utils.DeviceCandidate(
            "xpu:0", "XPU", torch.device("xpu:0"), "xpu"
        )
        expectations = {
            "auto": (["scalar"], ["scalar", "tiled"], ["scalar", "tiled"]),
            "scalar": (["scalar"], ["scalar"], ["scalar"]),
            "tiled": ([], ["tiled"], ["tiled"]),
        }
        for mode, (cpu_variants, cuda_variants, xpu_variants) in expectations.items():
            with self.subTest(mode=mode), mock.patch.object(
                network_injection, "EXECUTION_MODE", mode
            ), mock.patch.object(
                network_injection,
                "tiled_accelerator_support",
                return_value=(True, "supported"),
            ):
                self.assertEqual(
                    network_injection._execution_variants(cpu), cpu_variants
                )
                self.assertEqual(
                    network_injection._execution_variants(cuda), cuda_variants
                )
                self.assertEqual(
                    network_injection._execution_variants(xpu), xpu_variants
                )

    def test_forced_tiled_injection_rejects_missing_accelerator_before_benchmark(self):
        cpu = network_injection.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        with mock.patch.object(
            network_injection, "EXECUTION_MODE", "tiled"
        ), mock.patch.object(
            network_injection.Hardware_Utils,
            "get_available_devices",
            return_value=[cpu],
        ), self.assertRaisesRegex(ValueError, "no compatible CUDA/ROCm or XPU"):
            network_injection._benchmark_injection_plans(
                [(0, 1, "a", "b")],
                workers=1,
                input_h5="unused.h5",
                store=mock.Mock(),
                lengths=[2, 2],
                matmul_precision="ieee_fp32",
            )

    def test_first_batch_trials_run_each_feasible_configuration_once(self):
        cpu = network_injection.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        cuda = network_injection.Hardware_Utils.DeviceCandidate(
            "cuda:0", "CUDA", torch.device("cuda:0"), "cuda"
        )
        tasks = [(0, column, "a", f"h{column}") for column in range(1, 7)]
        memory = mock.Mock(
            free_bytes=12 << 30,
            total_bytes=16 << 30,
            matrix_bytes=8 << 30,
        )
        estimate = mock.Mock(
            feasible=True,
            projected_peak_bytes=10 << 30,
            safe_peak_bytes=13 << 30,
            reason="within reserved-VRAM boundary",
        )

        def complete_benchmark(*_args, **kwargs):
            trial = kwargs["benchmark_trial"]
            trial.start()
            trial.submitted = trial.completed = 6
            trial.stop(6)
            return []

        with mock.patch.object(
            network_injection, "DEVICE_SELECTION", "auto"
        ), mock.patch.object(
            network_injection.Hardware_Utils,
            "get_available_devices",
            return_value=[cpu, cuda],
        ), mock.patch.object(
            network_injection.Hardware_Utils, "release_device_cache"
        ), mock.patch.object(
            network_injection, "tiled_accelerator_support",
            return_value=(True, "supported"),
        ), mock.patch.object(
            network_injection, "_lane_candidates", return_value=[1, 2]
        ), mock.patch.object(
            network_injection, "cuda_memory_plan", return_value=memory
        ), mock.patch.object(
            network_injection, "estimate_cuda_working_set", return_value=estimate
        ), mock.patch.object(
            network_injection,
            "_execute_injection_plan",
            side_effect=complete_benchmark,
        ) as execute, redirect_stdout(io.StringIO()):
            plans = network_injection._benchmark_injection_plans(
                tasks,
                workers=2,
                input_h5="unused.h5",
                store=mock.Mock(),
                lengths=[2] * 7,
                matmul_precision="ieee_fp32",
            )

        self.assertTrue(plans)
        self.assertEqual(execute.call_count, 5)
        self.assertEqual([len(call.args[1]) for call in execute.call_args_list], [6, 6, 6, 6, 6])
        self.assertEqual(
            [call.args[1] for call in execute.call_args_list],
            [tasks] * 5,
        )

    def test_benchmark_skips_infeasible_fused_gpu_tiled_plans_under_tight_memory(self):
        # HI-4: build_adaptive_tile_plans(..., fused_gpu=True) drops infeasible lane plans.
        # Missing lanes under non-None memory_info must be skipped and excluded from
        # feasible_count instead of being treated as feasible and bypassing the VRAM gate.
        cuda = network_injection.Hardware_Utils.DeviceCandidate(
            "cuda:0", "CUDA", torch.device("cuda:0"), "cuda"
        )
        tasks = [(0, column, "s0", f"s{column}") for column in range(1, 5)]
        lengths = [64] * 5
        scalar_estimate = mock.Mock(
            feasible=True,
            projected_peak_bytes=1 << 28,
            safe_peak_bytes=1 << 29,
            reason="within reserved-VRAM boundary",
        )
        fused_lane1_estimate = mock.Mock(
            feasible=True,
            projected_peak_bytes=1 << 28,
            safe_peak_bytes=1 << 29,
            reason="within reserved-VRAM boundary",
        )
        fused_lane1_plan = mock.Mock(lanes=1, estimate=fused_lane1_estimate)

        def complete_benchmark(*_args, **kwargs):
            trial = kwargs["benchmark_trial"]
            trial.start()
            trial.submitted = trial.completed = len(tasks)
            trial.stop(len(tasks))
            return []

        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, "embs.h5")
            headers = [f"s{i}" for i in range(5)]
            with h5py.File(h5_path, "w") as hf:
                grp = hf.create_group("embeddings")
                for h, L in zip(headers, lengths):
                    grp.create_dataset(h, data=np.ones((L, 64), dtype=np.float32))
            store = network_injection.EmbeddingTileStore(h5_path, headers, host_cache_setting=0)

            # Subcase 1: real build_adaptive_tile_plans drops ALL fused tiled lanes under tight VRAM.
            tight_memory = mock.Mock(
                free_bytes=256 << 20,
                total_bytes=256 << 20,
                matrix_bytes=64 << 20,
            )
            stdout_all_dropped = io.StringIO()
            with mock.patch.object(
                network_injection, "DEVICE_SELECTION", "auto"
            ), mock.patch.object(
                network_injection.Hardware_Utils,
                "get_available_devices",
                return_value=[cuda],
            ), mock.patch.object(
                network_injection.Hardware_Utils, "release_device_cache"
            ), mock.patch.object(
                network_injection.Hardware_Utils, "begin_gpu_trial", return_value=None
            ), mock.patch.object(
                network_injection, "tiled_accelerator_support",
                return_value=(True, "supported"),
            ), mock.patch.object(
                network_injection.Align_GPU_Kernels, "align_kernels", return_value=object()
            ), mock.patch.object(
                network_injection, "_lane_candidates", return_value=[1, 2]
            ), mock.patch.object(
                network_injection, "cuda_memory_plan", return_value=tight_memory
            ), mock.patch.object(
                network_injection, "estimate_cuda_working_set", return_value=scalar_estimate
            ), mock.patch.object(
                network_injection, "_execute_injection_plan", side_effect=complete_benchmark
            ) as execute, redirect_stdout(stdout_all_dropped):
                ranked = network_injection._benchmark_injection_plans(
                    tasks,
                    workers=2,
                    input_h5=h5_path,
                    store=store,
                    lengths=lengths,
                    matmul_precision="ieee_fp32",
                )

            out_text = stdout_all_dropped.getvalue()
            executed_plans = [(c.args[0][1], c.args[0][2]) for c in execute.call_args_list]
            self.assertTrue(ranked)
            self.assertNotIn(("tiled", 1), executed_plans)
            self.assertNotIn(("tiled", 2), executed_plans)
            self.assertIn(("scalar", 1), executed_plans)
            self.assertIn(f"[Hardware] {cuda.display_name}: 2 feasible configurations.", out_text)
            self.assertEqual(out_text.count("skipped: no memory-safe tile plan"), 2)

            # Subcase 2: only lanes=1 fused plan is feasible; lanes=2 is dropped and must be skipped.
            stdout_partial = io.StringIO()
            with mock.patch.object(
                network_injection, "DEVICE_SELECTION", "auto"
            ), mock.patch.object(
                network_injection.Hardware_Utils,
                "get_available_devices",
                return_value=[cuda],
            ), mock.patch.object(
                network_injection.Hardware_Utils, "release_device_cache"
            ), mock.patch.object(
                network_injection.Hardware_Utils, "begin_gpu_trial", return_value=None
            ), mock.patch.object(
                network_injection, "tiled_accelerator_support",
                return_value=(True, "supported"),
            ), mock.patch.object(
                network_injection.Align_GPU_Kernels, "align_kernels", return_value=object()
            ), mock.patch.object(
                network_injection, "_lane_candidates", return_value=[1, 2]
            ), mock.patch.object(
                network_injection, "cuda_memory_plan", return_value=tight_memory
            ), mock.patch.object(
                network_injection, "build_adaptive_tile_plans", return_value=[fused_lane1_plan]
            ), mock.patch.object(
                network_injection, "estimate_cuda_working_set", return_value=scalar_estimate
            ), mock.patch.object(
                network_injection, "_execute_injection_plan", side_effect=complete_benchmark
            ) as execute_partial, redirect_stdout(stdout_partial):
                ranked_partial = network_injection._benchmark_injection_plans(
                    tasks,
                    workers=2,
                    input_h5=h5_path,
                    store=store,
                    lengths=lengths,
                    matmul_precision="ieee_fp32",
                )

            out_partial = stdout_partial.getvalue()
            executed_partial = [(c.args[0][1], c.args[0][2]) for c in execute_partial.call_args_list]
            self.assertTrue(ranked_partial)
            self.assertIn(("tiled", 1), executed_partial)
            self.assertNotIn(("tiled", 2), executed_partial)
            self.assertIn(f"[Hardware] {cuda.display_name}: 3 feasible configurations.", out_partial)
            self.assertEqual(out_partial.count("skipped: no memory-safe tile plan"), 1)

    def test_input_path_configuration_preserves_none_as_unselected(self):
        with mock.patch.object(network_injection, "OLD_NETWORK", None), \
                mock.patch.object(network_injection, "NEW_EMBEDDINGS", None):
            with self.assertRaisesRegex(ValueError, "existing network file"):
                network_injection.configure_input_paths()

    def test_input_path_configuration_does_not_open_selected_files(self):
        old_path = os.path.abspath(
            os.path.join("temporary", "old_[test-model]_network.h5")
        )
        new_path = os.path.abspath(
            os.path.join("temporary", "new_[test-model]_embeddings.h5")
        )
        with mock.patch.object(network_injection, "OLD_NETWORK", old_path), \
                mock.patch.object(network_injection, "NEW_EMBEDDINGS", new_path), \
                mock.patch.object(network_injection, "validate_network_schema") as validate:
            network_injection.configure_input_paths()
            validate.assert_not_called()
            self.assertEqual(network_injection.OLD_NETWORK, old_path)
            self.assertEqual(network_injection.NEW_EMBEDDINGS, new_path)

    def test_embedding_metadata_loader_rejects_unfinalized_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, "incomplete.h5")
            with h5py.File(h5_path, "w") as hf:
                Embedding_HDF5.create_metadata_first_file(
                    hf,
                    ["first"],
                    ["ACD"],
                    "test-model",
                    "float32",
                )

            with self.assertRaisesRegex(
                network_injection.EmbeddingFileError,
                "Embedding generation is incomplete",
            ):
                network_injection.load_embedding_metadata(h5_path)

    def test_embedding_metadata_loader_rejects_missing_embedding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, "missing_embedding.h5")
            with h5py.File(h5_path, "w") as hf:
                embeddings = Embedding_HDF5.create_metadata_first_file(
                    hf,
                    ["present", "missing"],
                    ["AC", "ACD"],
                    "test-model",
                    "float32",
                )
                embeddings.create_dataset(
                    "present",
                    data=np.ones((2, 4), dtype=np.float32),
                )
                hf.attrs["generation_complete"] = True

            with self.assertRaisesRegex(
                network_injection.EmbeddingFileError,
                "Embedding database is missing dataset 'missing'",
            ):
                network_injection.load_embedding_metadata(h5_path)

    def test_embedding_metadata_loader_preserves_order_and_lengths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, "complete.h5")
            with h5py.File(h5_path, "w") as hf:
                embeddings = Embedding_HDF5.create_metadata_first_file(
                    hf,
                    ["second_header", "first"],
                    ["AC", "ACD"],
                    "test-model",
                    "float32",
                )
                embeddings.create_dataset(
                    "first",
                    data=np.ones((3, 4), dtype=np.float32),
                )
                embeddings.create_dataset(
                    "second_header",
                    data=np.ones((2, 4), dtype=np.float32),
                )
                Embedding_HDF5.mark_generation_complete(hf)

            headers, safe_headers, lengths, manifest = (
                network_injection.load_embedding_metadata(h5_path)
            )

        self.assertEqual(headers, ["second_header", "first"])
        self.assertEqual(safe_headers, ["second_header", "first"])
        self.assertEqual(lengths, [2, 3])
        self.assertEqual(manifest.model_name, "test-model")

    def test_injection_stops_before_hashing_incomplete_embeddings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, "incomplete.h5")
            with h5py.File(h5_path, "w") as hf:
                Embedding_HDF5.create_metadata_first_file(
                    hf,
                    ["first"],
                    ["ACD"],
                    "test-model",
                    "float32",
                )

            output = io.StringIO()
            # set_start_method would switch this whole test process to spawn
            # on Linux; the run also assigns the sequence-set names.
            with mock.patch.object(
                network_injection,
                "NEW_EMBEDDINGS",
                h5_path,
            ), mock.patch.object(network_injection, "OLD_NETWORK", "unused.h5"), mock.patch.object(
                network_injection,
                "calculate_file_hash",
            ) as calculate_hash, mock.patch.multiple(
                network_injection,
                set_start_method=mock.DEFAULT,
                _old_seq_set=network_injection._old_seq_set,
                _new_seq_set=network_injection._new_seq_set,
            ), redirect_stdout(output):
                exit_code = network_injection.run_injection()

        self.assertEqual(exit_code, 1)
        calculate_hash.assert_not_called()
        self.assertIn("Cannot start Network Injection", output.getvalue())
        self.assertIn("Embedding generation is incomplete", output.getvalue())

    def test_alignment_worker_returns_scores_without_path_payload(self):
        matrix = np.array(
            [
                [3.0, -1.0],
                [-1.0, 3.0],
            ],
            dtype=np.float32,
        )

        with mock.patch.object(network_injection, "LOCAL_GAP_P", -2.0), mock.patch.object(network_injection, "GLOBAL_GAP_P", 0.0):
            result = network_injection.calculate_alignment_data((4, 7, matrix))

        self.assertEqual(len(result), 6)
        self.assertEqual(result[:2], (4, 7))
        self.assertEqual(float(result[2]), 2.0)
        self.assertEqual(int(result[3]), 2)
        self.assertEqual(float(result[4]), 6.0)
        self.assertEqual(int(result[5]), 2)

    def test_cpu_benchmark_drains_warmup_and_reuses_one_pool(self):
        events = []

        class FakePool:
            instances = 0

            def __init__(self, processes, initializer, initargs):
                type(self).instances += 1
                events.append("pool-open")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                events.append("pool-close")
                return False

            def map_async(self, function, chunk, chunksize):
                def get():
                    events.extend(f"pair-{task[1]}" for task in chunk)
                    return [function(task) for task in chunk]

                return SimpleNamespace(get=get)

        tasks = [(0, column, "a", f"h{column}") for column in range(1, 5)]
        trial = RecordingTrial(events)
        with mock.patch.object(
            network_injection, "Pool", FakePool
        ), mock.patch.object(
            network_injection,
            "calculate_cpu_pair",
            side_effect=lambda task: (task[0], task[1], 1.0, 1, 2.0, 1),
        ):
            results = network_injection.process_cpu_tasks(
                tasks,
                workers=2,
                input_h5="unused.h5",
                batch_id=-1,
                show_progress=False,
                benchmark_trial=trial,
            )

        # Four pairs are fewer than the warm-up minimum, so the warm-up runs all
        # of them and the timed phase starts again from the first pair.
        pairs = [f"pair-{column}" for column in range(1, 5)]
        self.assertEqual(FakePool.instances, 1)
        self.assertEqual(
            events,
            ["pool-open", *pairs, "trial-start", *pairs, "trial-stop", "pool-close"],
        )
        # Only the timed phase's results are returned.
        self.assertEqual(
            sorted(results),
            [(0, column, 1.0, 1, 2.0, 1) for column in range(1, 5)],
        )
        self.assertEqual((trial.submitted, trial.completed), (4, 4))

    def test_process_batch_streams_benchmarked_plans_into_one_batch_file(self):
        # run_injection always passes the embedding store, the plan's lanes,
        # variant and the inherited precision; both accelerator variants must
        # stream their results into the atomic batch writer.
        tasks = [(0, 2, "a", "c"), (1, 2, "b", "c")]
        expected = [(0, 2, 1.5, 3, 2.5, 4), (1, 2, 3.5, 5, 4.5, 6)]
        store = mock.Mock()
        plan = mock.Mock(matrix_bytes=64 << 20)

        def stream_results(*_args, **kwargs):
            kwargs["result_callback"](list(expected))
            return []

        for variant in ("scalar", "tiled"):
            with self.subTest(variant=variant), \
                    tempfile.TemporaryDirectory() as temp_dir, \
                    mock.patch.object(
                        network_injection, "RESULTS_DIR", temp_dir
                    ), mock.patch.object(
                        network_injection,
                        "get_accelerator_backend",
                        return_value=mock.Mock(device_type="cuda"),
                    ), mock.patch.object(
                        network_injection, "cuda_memory_plan", return_value=plan
                    ) as memory_plan, mock.patch.object(
                        network_injection,
                        "_run_accelerated_pipeline",
                        side_effect=stream_results,
                    ) as scalar, mock.patch.object(
                        network_injection,
                        "run_tiled_accelerator_pipeline",
                        side_effect=stream_results,
                    ) as tiled, mock.patch.object(
                        network_injection, "process_cpu_tasks"
                    ) as cpu:
                network_injection.process_batch(
                    tasks,
                    batch_id=5,
                    workers=4,
                    new_emb_path="unused.h5",
                    embedding_checksum="checksum",
                    model_name="test-model",
                    saving_mode="float16",
                    gap_penalties=[-2.0, 0.0],
                    device=torch.device("cuda:0"),
                    accelerator_workers=2,
                    execution_variant=variant,
                    matmul_precision="tf32",
                    embedding_store=store,
                    sequence_lengths=[3, 5, 7],
                )

                cpu.assert_not_called()
                if variant == "scalar":
                    tiled.assert_not_called()
                    scalar.assert_called_once()
                    self.assertEqual(
                        scalar.call_args.args,
                        (tasks, 4, "unused.h5", torch.device("cuda:0"), 5, 2, True),
                    )
                    self.assertEqual(scalar.call_args.kwargs["matmul_precision"], "tf32")
                    writer = scalar.call_args.kwargs["result_callback"]
                else:
                    scalar.assert_not_called()
                    tiled.assert_called_once()
                    memory_plan.assert_called_once_with(torch.device("cuda:0"), lanes=2)
                    kwargs = tiled.call_args.kwargs
                    self.assertIs(kwargs["store"], store)
                    self.assertEqual(kwargs["lengths"], [3, 5, 7])
                    self.assertEqual(kwargs["lanes"], 2)
                    self.assertEqual(kwargs["precision"], "tf32")
                    self.assertEqual(kwargs["matrix_budget_override"], 64 << 20)
                    self.assertIs(
                        kwargs["alignment_callback"],
                        network_injection.calculate_alignment_data,
                    )
                    self.assertIs(
                        kwargs["batch_alignment_callback"],
                        network_injection.calculate_alignment_batch,
                    )
                    writer = kwargs["result_callback"]
                self.assertIsInstance(writer, network_injection._PartialBatchWriter)

                output_path = os.path.join(temp_dir, "batch_00005.h5")
                self.assertEqual(os.listdir(temp_dir), ["batch_00005.h5"])
                with h5py.File(output_path, "r") as hf:
                    self.assertEqual(hf.attrs["embedding_checksum"], "checksum")
                    self.assertEqual(hf.attrs["model_name"], "test-model")
                    self.assertEqual(hf.attrs["saving_mode"], "float16")
                    self.assertEqual(hf.attrs["matmul_precision"], "tf32")
                    np.testing.assert_array_equal(hf["i"][:], [0, 1])
                    np.testing.assert_array_equal(hf["j"][:], [2, 2])
                    np.testing.assert_array_equal(hf["l_score"][:], [1.5, 3.5])
                    np.testing.assert_array_equal(hf["g_len"][:], [4, 6])

    def test_partial_writer_records_tf32_and_publishes_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "batch_00000.h5")
            writer = network_injection._PartialBatchWriter(
                output_path,
                "checksum",
                "test-model",
                "float16",
                [-2.0, 0.0],
                "tf32",
            )
            writer([(0, 1, 1.0, 1, 2.0, 1)])
            self.assertFalse(os.path.exists(output_path))
            self.assertTrue(os.path.exists(output_path + ".partial"))
            writer.publish()
            self.assertTrue(os.path.exists(output_path))
            self.assertFalse(os.path.exists(output_path + ".partial"))
            with h5py.File(output_path, "r") as hf:
                self.assertEqual(hf.attrs["matmul_precision"], "tf32")
                self.assertEqual(len(hf["i"]), 1)

    def test_legacy_batch_precision_is_ieee_fp32(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_path = os.path.join(temp_dir, "batch_00000.h5")
            with h5py.File(batch_path, "w") as hf:
                hf.attrs["embedding_checksum"] = "checksum"
                hf.attrs["model_name"] = "test-model"
                hf.attrs["saving_mode"] = "float32"
                hf.attrs["gap_penalties"] = np.asarray([-2.0, 0.0], np.float32)
                for name, values in {
                    "i": [0], "j": [1], "l_score": [1.0], "l_len": [1],
                    "g_score": [2.0], "g_len": [1],
                }.items():
                    hf.create_dataset(name, data=values)
            with mock.patch.object(network_injection, "RESULTS_DIR", temp_dir):
                computed = network_injection.scan_existing_batches(
                    2,
                    "checksum",
                    "test-model",
                    "float32",
                    [-2.0, 0.0],
                    "ieee_fp32",
                )
            self.assertEqual(computed, {1})

    # s0-s2 come from the old network; s3 and s4 are new sequences.
    COMPILE_OLD_PAIRS = [(0, 1), (0, 2), (1, 2)]
    COMPILE_BATCHES = {
        "batch_00000.h5": [(0, 3), (1, 3), (2, 3)],
        "batch_00001.h5": [(0, 4), (1, 4), (2, 4)],
        "batch_00002.h5": [(3, 4)],
    }

    def _write_compile_batches(self, results_dir, corrupt=None, corruption=None):
        os.makedirs(results_dir)
        for name, pairs in self.COMPILE_BATCHES.items():
            path = os.path.join(results_dir, name)
            if name == corrupt and corruption == "not_hdf5":
                with open(path, "wb") as handle:
                    handle.write(b"truncated batch")
                continue
            arr_i = np.array([pair[0] for pair in pairs], dtype=np.uint32)
            arr_j = np.array([pair[1] for pair in pairs], dtype=np.uint32)
            columns = {
                "i": arr_i,
                "j": arr_j,
                "l_score": (arr_i * 10 + arr_j + 0.5).astype(np.float32),
                "l_len": (arr_i + arr_j + 1).astype(np.uint16),
                "g_score": (arr_i * 10 + arr_j + 0.25).astype(np.float32),
                "g_len": (arr_i + arr_j + 2).astype(np.uint16),
            }
            if name == corrupt and corruption.startswith("missing_"):
                del columns[corruption[len("missing_"):]]
            with h5py.File(path, "w") as hf:
                for column, data in columns.items():
                    hf.create_dataset(column, data=data)

    def _compile_batches(self, temp_dir):
        new_n = 5
        output = os.path.join(temp_dir, "network", "set_[model]_network.h5")
        old_count = len(self.COMPILE_OLD_PAIRS)
        stdout = io.StringIO()
        with mock.patch.object(
            network_injection,
            "RESULTS_DIR",
            os.path.join(temp_dir, "batches"),
        ), mock.patch.object(
            network_injection,
            "FINAL_OUTPUT_NET",
            output,
        ), redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            network_injection.compile_final_output(
                new_headers=[f"s{index}" for index in range(new_n)],
                seq_lens=[5] * new_n,
                required_pairs={
                    i * new_n + j
                    for i in range(new_n)
                    for j in range(i + 1, new_n)
                },
                cached_old_pairs={
                    i * new_n + j for i, j in self.COMPILE_OLD_PAIRS
                },
                new_N=new_n,
                old_header_to_idx={"s0": 0, "s1": 1, "s2": 2},
                old_l_score=np.arange(old_count, dtype=np.float32) + 100.5,
                old_l_len=np.arange(old_count, dtype=np.uint16) + 101,
                old_g_score=np.arange(old_count, dtype=np.float32) + 100.25,
                old_g_len=np.arange(old_count, dtype=np.uint16) + 102,
                actual_idx_map=np.arange(old_count),
                current_checksum="checksum",
                model_name="model",
                saving_mode="float32",
                gap_penalties=[-2.0, 0.0],
            )
        return output, stdout.getvalue()

    def test_compile_publishes_complete_network_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_compile_batches(os.path.join(temp_dir, "batches"))
            output, stdout = self._compile_batches(temp_dir)

            self.assertEqual(
                os.listdir(os.path.dirname(output)),
                [os.path.basename(output)],
            )
            self.assertIn("✅ Compilation complete!", stdout)
            old_rows = {
                pair: index for index, pair in enumerate(self.COMPILE_OLD_PAIRS)
            }
            pairs = sorted(
                list(old_rows)
                + [pair for batch in self.COMPILE_BATCHES.values() for pair in batch]
            )
            expected_l_score = [
                100.5 + old_rows[pair] if pair in old_rows
                else pair[0] * 10 + pair[1] + 0.5
                for pair in pairs
            ]
            expected_g_len = [
                102 + old_rows[pair] if pair in old_rows
                else pair[0] + pair[1] + 2
                for pair in pairs
            ]
            with h5py.File(output, "r") as hf:
                np.testing.assert_array_equal(hf["i"][:], [p[0] for p in pairs])
                np.testing.assert_array_equal(hf["j"][:], [p[1] for p in pairs])
                np.testing.assert_array_equal(hf["l_score"][:], expected_l_score)
                np.testing.assert_array_equal(hf["g_len"][:], expected_g_len)

    def test_pair_ids_above_65535_nodes_do_not_wrap_in_32_bits(self):
        # uint32 indices times new_N wrapped modulo 2**32 for a network over 65,536 nodes.
        new_n = 100000
        pairs = [(70000, 70001), (3, 99999)]
        wrong_id = (70000 * new_n + 70001) % 2**32
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = os.path.join(temp_dir, "batches")
            os.makedirs(results_dir)
            arr_i = np.array([pair[0] for pair in pairs], dtype=np.uint32)
            arr_j = np.array([pair[1] for pair in pairs], dtype=np.uint32)
            with h5py.File(os.path.join(results_dir, "batch_00000.h5"), "w") as hf:
                for key, value in {
                    "embedding_checksum": "checksum",
                    "model_name": "model",
                    "saving_mode": "float32",
                    "gap_penalties": np.array([-2.0, 0.0], np.float32),
                    "matmul_precision": "ieee_fp32",
                }.items():
                    hf.attrs[key] = value
                hf.create_dataset("i", data=arr_i)
                hf.create_dataset("j", data=arr_j)
                hf.create_dataset("l_score", data=np.array([1.5, 2.5], np.float32))
                hf.create_dataset("l_len", data=np.array([11, 12], np.uint16))
                hf.create_dataset("g_score", data=np.array([3.5, 4.5], np.float32))
                hf.create_dataset("g_len", data=np.array([13, 14], np.uint16))
            output = os.path.join(temp_dir, "network", "set_[model]_network.h5")
            true_ids = {i * new_n + j for i, j in pairs}
            self.assertNotIn(wrong_id, true_ids)
            with mock.patch.object(
                network_injection, "RESULTS_DIR", results_dir
            ), mock.patch.object(
                network_injection, "FINAL_OUTPUT_NET", output
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("error")
                computed = network_injection.scan_existing_batches(
                    new_n, "checksum", "model", "float32", [-2.0, 0.0], "ieee_fp32"
                )
                self.assertEqual(computed, true_ids)
                # A required pair that the wrapped id would have hit instead.
                decoy = (wrong_id // new_n, wrong_id % new_n)
                required = true_ids | {decoy[0] * new_n + decoy[1]}
                network_injection.compile_final_output(
                    new_headers=[f"s{index}" for index in range(new_n)],
                    seq_lens=[5] * new_n,
                    required_pairs=required,
                    cached_old_pairs=set(),
                    new_N=new_n,
                    old_header_to_idx={},
                    old_l_score=np.zeros(0, np.float32),
                    old_l_len=np.zeros(0, np.uint16),
                    old_g_score=np.zeros(0, np.float32),
                    old_g_len=np.zeros(0, np.uint16),
                    actual_idx_map=np.zeros(0, np.int64),
                    current_checksum="checksum",
                    model_name="model",
                    saving_mode="float32",
                    gap_penalties=[-2.0, 0.0],
                )
            with h5py.File(output, "r") as hf:
                rows = {
                    (int(i), int(j)): (float(l), float(g))
                    for i, j, l, g in zip(
                        hf["i"][:], hf["j"][:], hf["l_score"][:], hf["g_score"][:]
                    )
                }
        self.assertEqual(rows[(70000, 70001)], (1.5, 3.5))
        self.assertEqual(rows[(3, 99999)], (2.5, 4.5))
        # The pair the wrapped id pointed at was not computed, so it stays at zero.
        self.assertEqual(rows[decoy], (0.0, 0.0))
        self.assertEqual(len(rows), 3)

    def test_corrupt_batch_fails_compilation_without_a_network_file(self):
        # Unreadable batches, and batches missing a score column, used to be
        # skipped with a warning: their pairs stayed in the network with zero
        # scores and lengths, and the job reported success. A batch missing
        # i/j was skipped without even a warning.
        cases = (
            ("batch_00001.h5", "not_hdf5"),
            ("batch_00001.h5", "missing_l_score"),
            ("batch_00002.h5", "missing_g_score"),
            ("batch_00000.h5", "missing_i"),
        )
        for corrupt, corruption in cases:
            with self.subTest(batch=corrupt, corruption=corruption):
                with tempfile.TemporaryDirectory() as temp_dir:
                    self._write_compile_batches(
                        os.path.join(temp_dir, "batches"),
                        corrupt,
                        corruption,
                    )
                    output = os.path.join(
                        temp_dir, "network", "set_[model]_network.h5"
                    )
                    with self.assertRaisesRegex(
                        network_injection.NetworkCompilationError,
                        f"Error reading .*{corrupt} during compilation",
                    ):
                        self._compile_batches(temp_dir)

                    self.assertFalse(os.path.exists(output))
                    self.assertFalse(os.path.exists(output + ".partial"))

    def test_failed_network_write_keeps_previous_network(self):
        real_create_dataset = h5py.Group.create_dataset

        def disk_full_on_last_dataset(group, name, *args, **kwargs):
            if name == "g_len":
                raise OSError("No space left on device")
            return real_create_dataset(group, name, *args, **kwargs)

        for previous in (None, b"previous complete network"):
            with self.subTest(previous_network=previous is not None):
                with tempfile.TemporaryDirectory() as temp_dir:
                    self._write_compile_batches(os.path.join(temp_dir, "batches"))
                    output = os.path.join(
                        temp_dir, "network", "set_[model]_network.h5"
                    )
                    if previous is not None:
                        os.makedirs(os.path.dirname(output))
                        with open(output, "wb") as handle:
                            handle.write(previous)

                    with mock.patch.object(
                        h5py.Group,
                        "create_dataset",
                        disk_full_on_last_dataset,
                    ), self.assertRaisesRegex(OSError, "No space left"):
                        self._compile_batches(temp_dir)

                    self.assertEqual(
                        os.listdir(os.path.dirname(output)),
                        [] if previous is None else [os.path.basename(output)],
                    )
                    if previous is not None:
                        with open(output, "rb") as handle:
                            self.assertEqual(handle.read(), previous)

    def test_main_exits_nonzero_when_a_batch_cannot_be_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_compile_batches(
                os.path.join(temp_dir, "batches"),
                "batch_00001.h5",
                "missing_l_score",
            )
            stdout = io.StringIO()
            with mock.patch.object(
                network_injection,
                "load_tool_settings",
            ), mock.patch.object(
                network_injection,
                "run_injection",
                side_effect=lambda: self._compile_batches(temp_dir),
            ), redirect_stdout(stdout):
                exit_code = network_injection.main([])

            self.assertEqual(exit_code, 1)
            self.assertIn("❌ Cannot compile the final network", stdout.getvalue())
            self.assertIn("batch_00001.h5", stdout.getvalue())
            self.assertFalse(
                os.path.exists(
                    os.path.join(temp_dir, "network", "set_[model]_network.h5")
                )
            )

    def _write_injection_inputs(self, temp_dir):
        embed_dir = os.path.join(temp_dir, "embed")
        network_dir = os.path.join(temp_dir, "network")
        os.makedirs(embed_dir)
        os.makedirs(network_dir)
        sequences = {"old_a": "ACDE", "old_b": "ACDF", "new_c": "GHIKL"}
        rng = np.random.default_rng(7)
        embeddings_path = os.path.join(
            embed_dir, "newset_[esm2_t6_8m]_embeddings.h5"
        )
        with h5py.File(embeddings_path, "w") as hf:
            group = Embedding_HDF5.create_metadata_first_file(
                hf,
                list(sequences),
                list(sequences.values()),
                "esm2_t6_8m",
                "float32",
            )
            for header, sequence in sequences.items():
                group.create_dataset(
                    header,
                    data=rng.standard_normal((len(sequence), 4)).astype(np.float32),
                )
            Embedding_HDF5.mark_generation_complete(hf)

        network_path = os.path.join(network_dir, "oldset_[esm2_t6_8m]_network.h5")
        with h5py.File(network_path, "w") as hf:
            hf.attrs["model_name"] = "esm2_t6_8m"
            hf.attrs["saving_mode"] = "float32"
            hf.attrs["gap_penalties"] = np.asarray([-2.0, 0.0], np.float32)
            hf.attrs["matmul_precision"] = "ieee_fp32"
            hf.create_dataset(
                "headers",
                data=np.array(["old_a", "old_b"], dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            hf.create_dataset("seq_lens", data=np.asarray([4, 4], np.uint16))
            for name, data in {
                "i": np.asarray([0], np.uint16),
                "j": np.asarray([1], np.uint16),
                "l_score": np.asarray([1.5], np.float32),
                "l_len": np.asarray([4], np.uint16),
                "g_score": np.asarray([2.5], np.float32),
                "g_len": np.asarray([4], np.uint16),
            }.items():
                hf.create_dataset(name, data=data)
        return embed_dir, network_dir

    def test_existing_output_precision_is_checked_at_the_output_path(self):
        # The check used to run before the output path was configured, so a
        # fresh process called os.path.exists(None) and every run stopped
        # there with TypeError.
        with tempfile.TemporaryDirectory() as temp_dir:
            embed_dir, network_dir = self._write_injection_inputs(temp_dir)
            existing = os.path.join(network_dir, "newset_[esm2_t6_8m]_network.h5")
            with h5py.File(existing, "w") as hf:
                hf.attrs["matmul_precision"] = "bf16"
            with open(existing, "rb") as handle:
                existing_bytes = handle.read()

            with mock.patch.multiple(
                network_injection,
                OLD_NETWORK="oldset_[esm2_t6_8m]_network.h5",
                NEW_EMBEDDINGS="newset_[esm2_t6_8m]_embeddings.h5",
                EMBED_DIR=embed_dir,
                NETWORK_DIR=network_dir,
                FINAL_OUTPUT_NET=None,
                RESULTS_DIR=None,
                CONFIG_FILE=None,
                LOCAL_GAP_P=None,
                GLOBAL_GAP_P=None,
                _old_seq_set="UnknownOld",
                _new_seq_set="UnknownNew",
                _model_name="unknown",
                calculate_file_hash=mock.DEFAULT,
                set_start_method=mock.DEFAULT,
            ) as patched, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    network_injection.EmbeddingFileError,
                    "Completed output precision conflict",
                ):
                    network_injection.run_injection()

            patched["calculate_file_hash"].assert_not_called()
            with open(existing, "rb") as handle:
                self.assertEqual(handle.read(), existing_bytes)

    def test_cpu_workers_use_gap_penalties_inherited_from_network(self):
        # Spawned workers re-import this module, where both penalties stay
        # None unless the pool hands them over; numba then rejected the
        # alignment kernel call.
        rng = np.random.default_rng(11)
        embeddings = {
            name: rng.standard_normal((length, 4)).astype(np.float32)
            for name, length in (("a", 5), ("b", 7), ("c", 6))
        }
        tasks = [(0, 1, "a", "b"), (0, 2, "a", "c")]
        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, "embeddings.h5")
            with h5py.File(h5_path, "w") as hf:
                for name, data in embeddings.items():
                    hf.create_dataset(f"embeddings/{name}", data=data)

            with mock.patch.object(
                network_injection, "LOCAL_GAP_P", -2.0
            ), mock.patch.object(
                network_injection, "GLOBAL_GAP_P", -1.0
            ), mock.patch.dict(os.environ, {
                "SSN_TOOL_SETTINGS_SCRIPT": "Network_Injection.py",
                "SSN_TOOL_SETTINGS_FILE": os.path.join(
                    PROJECT_ROOT, "tests", "nonexistent-settings.json"
                ),
            }):
                expected = [
                    network_injection.calculate_alignment_data((
                        i,
                        j,
                        network_injection.compute_score_matrix_torch(
                            embeddings[h_i], embeddings[h_j], torch.device("cpu")
                        ),
                    ))
                    for i, j, h_i, h_j in tasks
                ]
                results = network_injection.process_cpu_tasks(
                    tasks,
                    workers=1,
                    input_h5=h5_path,
                    batch_id=0,
                    show_progress=False,
                )

        self.assertEqual(sorted(results), expected)

    def _run_main(self, temp_dir, embeddings=None, **patches):
        """Run main() on temporary inputs, settings loading and compile stubbed.

        ``embeddings`` may rewrite the valid embeddings file before the run.
        Returns the process exit status, stdout and the compile mock.
        """
        embed_dir, network_dir = self._write_injection_inputs(temp_dir)
        if embeddings is not None:
            embeddings(os.path.join(embed_dir, "newset_[esm2_t6_8m]_embeddings.h5"))
        # The job assigns these globals; patching them restores the originals.
        assigned = (
            "_old_seq_set",
            "_new_seq_set",
            "_model_name",
            "RESULTS_DIR",
            "FINAL_OUTPUT_NET",
            "CONFIG_FILE",
            "LOCAL_GAP_P",
            "GLOBAL_GAP_P",
        )
        values = {
            **{name: getattr(network_injection, name) for name in assigned},
            "OLD_NETWORK": "oldset_[esm2_t6_8m]_network.h5",
            "NEW_EMBEDDINGS": "newset_[esm2_t6_8m]_embeddings.h5",
            "EMBED_DIR": embed_dir,
            "NETWORK_DIR": network_dir,
            "EXECUTION_MODE": "auto",
            "DEVICE_SELECTION": "auto",
            "load_tool_settings": mock.DEFAULT,
            "set_start_method": mock.DEFAULT,
            "compile_final_output": mock.DEFAULT,
            **patches,
        }
        stdout = io.StringIO()
        with mock.patch.multiple(network_injection, **values) as patched, \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            try:
                exit_code = network_injection.main([])
            except SystemExit as request:
                # The interpreter prints a string code and exits with status 1.
                exit_code = 1 if isinstance(request.code, str) else request.code
                print(request.code)
        return exit_code, stdout.getvalue(), patched["compile_final_output"]

    def test_main_exits_nonzero_when_injection_cannot_start(self):
        # The MCP job runner reports exit code 0 as "succeeded", so the
        # "Cannot start" paths used to reach agents as successful jobs that
        # wrote nothing.
        def write_garbage(path):
            with open(path, "wb") as handle:
                handle.write(b"not an embeddings file")

        def write_incomplete(path):
            with h5py.File(path, "w") as hf:
                Embedding_HDF5.create_metadata_first_file(
                    hf, ["new_c"], ["GHIKL"], "esm2_t6_8m", "float32"
                )

        cases = (
            ("no network", {"OLD_NETWORK": None},
             "❌ Cannot start Network Injection:\n"
             "No existing network file has been selected."),
            ("no embeddings", {"NEW_EMBEDDINGS": None},
             "❌ Cannot start Network Injection:\n"
             "No new embeddings file has been selected."),
            ("unknown mode", {"EXECUTION_MODE": "vector"},
             "❌ Cannot start Network Injection:\nExecution mode must be"),
            ("tiled on cpu", {"EXECUTION_MODE": "tiled", "DEVICE_SELECTION": "cpu"},
             "❌ Cannot start Network Injection:\n"
             "Tiled execution requires a CUDA/ROCm or XPU accelerator"),
            ("invalid embeddings", {"embeddings": write_garbage},
             "❌ Cannot start Network Injection:\nEmbedding file '"),
            ("incomplete embeddings", {"embeddings": write_incomplete},
             ("❌ Cannot start Network Injection:\nEmbedding file '",
              "validation failed: Embedding generation is incomplete")),
            # These already exited with status 1 through sys.exit.
            ("missing embeddings", {"NEW_EMBEDDINGS": "absent.h5"},
             "❌ Error: New embeddings file not found"),
            ("missing network", {"OLD_NETWORK": "absent.h5"},
             "❌ Error: Old network file not found"),
        )
        for label, patches, message in cases:
            with self.subTest(label), tempfile.TemporaryDirectory() as temp_dir:
                exit_code, stdout, compile_output = self._run_main(
                    temp_dir, **patches
                )

                self.assertEqual(exit_code, 1)
                for fragment in (message,) if isinstance(message, str) else message:
                    self.assertIn(fragment, stdout)
                compile_output.assert_not_called()
                self.assertEqual(
                    sorted(os.listdir(os.path.join(temp_dir, "network"))),
                    ["oldset_[esm2_t6_8m]_network.h5"],
                )

    def test_main_exits_zero_when_injection_runs(self):
        plan = mock.Mock(variant="scalar", lanes=1)
        plan.candidate.is_cpu = True
        with tempfile.TemporaryDirectory() as temp_dir:
            exit_code, stdout, compile_output = self._run_main(
                temp_dir,
                _benchmark_injection_plans=mock.Mock(return_value=[plan]),
                process_batch=mock.DEFAULT,
            )

        self.assertEqual(exit_code, 0)
        self.assertNotIn("❌", stdout)
        self.assertIn("Pairs queued for calculation: 2", stdout)
        compile_output.assert_called_once()

    # Old network A, B, C; its rows are stored out of canonical pair order.
    REORDER_OLD_EDGES = {
        ("B", "C"): (21.5, 21, 22.5, 22),
        ("A", "B"): (11.5, 11, 12.5, 12),
        ("A", "C"): (31.5, 31, 32.5, 32),
    }
    # Scores the stubbed CPU workers return for the pairs with the new D.
    REORDER_NEW_EDGES = {
        ("C", "D"): (41.5, 41, 42.5, 42),
        ("A", "D"): (51.5, 51, 52.5, 52),
        ("B", "D"): (61.5, 61, 62.5, 62),
    }

    def test_reordered_sequence_set_keeps_each_old_edge_on_its_own_pair(self):
        # The new FASTA lists C, A, B, D: every reused old edge changes index
        # and two of them reverse their orientation (C now precedes A and B).
        # A wrong old-to-new index map would give an edge another pair's scores.
        old_sequences = {"A": "ACDE", "B": "ACDFG", "C": "ACDFGH"}
        new_sequences = {"C": "ACDFGH", "A": "ACDE", "B": "ACDFG", "D": "KLMNPQR"}
        old_index = {header: index for index, header in enumerate(old_sequences)}
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as temp_dir:
            embed_dir = os.path.join(temp_dir, "embed")
            network_dir = os.path.join(temp_dir, "network")
            os.makedirs(embed_dir)
            os.makedirs(network_dir)
            with h5py.File(
                os.path.join(embed_dir, "newset_[esm2_t6_8m]_embeddings.h5"), "w"
            ) as hf:
                group = Embedding_HDF5.create_metadata_first_file(
                    hf,
                    list(new_sequences),
                    list(new_sequences.values()),
                    "esm2_t6_8m",
                    "float32",
                )
                for header, sequence in new_sequences.items():
                    group.create_dataset(
                        header,
                        data=rng.standard_normal((len(sequence), 4)).astype(np.float32),
                    )
                Embedding_HDF5.mark_generation_complete(hf)
            with h5py.File(
                os.path.join(network_dir, "oldset_[esm2_t6_8m]_network.h5"), "w"
            ) as hf:
                hf.attrs["model_name"] = "esm2_t6_8m"
                hf.attrs["saving_mode"] = "float32"
                hf.attrs["gap_penalties"] = np.asarray([-2.0, 0.0], np.float32)
                hf.attrs["matmul_precision"] = "ieee_fp32"
                hf.create_dataset(
                    "headers",
                    data=np.array(list(old_sequences), dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                hf.create_dataset(
                    "seq_lens",
                    data=np.asarray([len(s) for s in old_sequences.values()], np.uint16),
                )
                edges = list(self.REORDER_OLD_EDGES.items())
                columns = list(zip(*(values for _pair, values in edges)))
                for name, data, dtype in (
                    ("i", [old_index[u] for (u, _v), _ in edges], np.uint16),
                    ("j", [old_index[v] for (_u, v), _ in edges], np.uint16),
                    ("l_score", columns[0], np.float32),
                    ("l_len", columns[1], np.uint16),
                    ("g_score", columns[2], np.float32),
                    ("g_len", columns[3], np.uint16),
                ):
                    hf.create_dataset(name, data=np.asarray(data, dtype))

            plan = mock.Mock(variant="scalar", lanes=1)
            plan.candidate.is_cpu = True
            plan.candidate.device = torch.device("cpu")

            def compute_new_pairs(tasks, *_args, **_kwargs):
                return [
                    (i, j, *self.REORDER_NEW_EDGES[(h_i, h_j)])
                    for i, j, h_i, h_j in tasks
                ]

            cpu_tasks = mock.Mock(side_effect=compute_new_pairs)
            # The job assigns these globals; patching them restores the originals.
            assigned = (
                "_old_seq_set", "_new_seq_set", "_model_name", "RESULTS_DIR",
                "FINAL_OUTPUT_NET", "CONFIG_FILE", "LOCAL_GAP_P", "GLOBAL_GAP_P",
            )
            with mock.patch.multiple(
                network_injection,
                **{name: getattr(network_injection, name) for name in assigned},
                OLD_NETWORK="oldset_[esm2_t6_8m]_network.h5",
                NEW_EMBEDDINGS="newset_[esm2_t6_8m]_embeddings.h5",
                EMBED_DIR=embed_dir,
                NETWORK_DIR=network_dir,
                EXECUTION_MODE="auto",
                DEVICE_SELECTION="auto",
                set_start_method=mock.DEFAULT,
                _benchmark_injection_plans=mock.Mock(return_value=[plan]),
                process_cpu_tasks=cpu_tasks,
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                exit_code = network_injection.run_injection()

            self.assertIsNone(exit_code)
            # Only the pairs with the new sequence D are calculated.
            cpu_tasks.assert_called_once()
            self.assertEqual(
                cpu_tasks.call_args.args[0],
                [(0, 3, "C", "D"), (1, 3, "A", "D"), (2, 3, "B", "D")],
            )
            new_headers = list(new_sequences)
            expected = [
                (i, j, *(
                    self.REORDER_NEW_EDGES.get((new_headers[i], new_headers[j]))
                    or self.REORDER_OLD_EDGES[tuple(sorted((new_headers[i], new_headers[j])))]
                ))
                for i in range(4)
                for j in range(i + 1, 4)
            ]
            output = os.path.join(network_dir, "newset_[esm2_t6_8m]_network.h5")
            with h5py.File(output, "r") as hf:
                self.assertEqual(
                    [header.decode() for header in hf["headers"][:]], new_headers
                )
                actual = list(zip(*(
                    hf[name][:].tolist()
                    for name in ("i", "j", "l_score", "l_len", "g_score", "g_len")
                )))
        # (0, 1) is C-A, so it carries A-C's old scores; (0, 2) carries B-C's
        # and (1, 2) A-B's.
        self.assertEqual(actual, expected)
        self.assertEqual(
            actual[:2],
            [(0, 1, 31.5, 31, 32.5, 32), (0, 2, 21.5, 21, 22.5, 22)],
        )

    def test_a_calibration_sample_does_not_lower_the_cut_for_new_pairs(self):
        # One-residue embeddings, so each pooled vector is the vector itself and
        # the lengths are equal. A-B and A-C are kept pairs (cosine about 0.98 and
        # up); C-D is the prefilter's calibration pair, far below its cut (about
        # -0.92). Counting it as the lowest kept edge would set s_min to -0.92.
        vectors = {
            "A": [1.0, 0.0], "B": [1.0, 0.1], "C": [1.0, 0.2], "D": [-1.0, 0.2], "E": [0.0, 1.0],
        }
        old_headers = ["A", "B", "C", "D"]
        with tempfile.TemporaryDirectory() as temp_dir:
            embed_dir = os.path.join(temp_dir, "embed")
            network_dir = os.path.join(temp_dir, "network")
            os.makedirs(embed_dir)
            os.makedirs(network_dir)
            with h5py.File(os.path.join(embed_dir, "newset_[esm2_t6_8m]_embeddings.h5"), "w") as hf:
                group = Embedding_HDF5.create_metadata_first_file(
                    hf, list(vectors), ["K"] * len(vectors), "esm2_t6_8m", "float32",
                )
                for header, vector in vectors.items():
                    group.create_dataset(header, data=np.asarray([vector], np.float32))
                Embedding_HDF5.mark_generation_complete(hf)
            with h5py.File(os.path.join(network_dir, "oldset_[esm2_t6_8m]_network.h5"), "w") as hf:
                hf.attrs["model_name"] = "esm2_t6_8m"
                hf.attrs["saving_mode"] = "float32"
                hf.attrs["gap_penalties"] = np.asarray([-2.0, 0.0], np.float32)
                hf.attrs["matmul_precision"] = "ieee_fp32"
                hf.create_dataset(
                    "headers", data=np.array(old_headers, dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                hf.create_dataset("seq_lens", data=np.ones(4, np.uint16))
                for name, data, dtype in (
                    ("i", [0, 0, 2], np.uint16), ("j", [1, 2, 3], np.uint16),
                    ("l_score", [9.0, 8.0, 1.0], np.float32), ("l_len", [1, 1, 1], np.uint16),
                    ("g_score", [9.0, 8.0, 1.0], np.float32), ("g_len", [1, 1, 1], np.uint16),
                ):
                    hf.create_dataset(name, data=np.asarray(data, dtype))
                Prefilter_Calibration.write_record(hf, Prefilter_Calibration.PrefilterRecord(
                    strength=50.0, threshold=0.9, pooling="max", length_power=2.0, kept=2, skipped=4,
                    sample=Prefilter_Calibration.CalibrationSample(
                        np.array([2]), np.array([3]), np.array([4.0])
                    ),
                ), np.uint16)

            cpu_tasks = mock.Mock(
                side_effect=lambda tasks, *_a, **_k: [(i, j, 1.0, 1, 1.0, 1) for i, j, *_ in tasks]
            )
            assigned = (
                "_old_seq_set", "_new_seq_set", "_model_name", "RESULTS_DIR",
                "FINAL_OUTPUT_NET", "CONFIG_FILE", "LOCAL_GAP_P", "GLOBAL_GAP_P",
            )
            plan = mock.Mock(variant="scalar", lanes=1)
            plan.candidate.is_cpu = True
            plan.candidate.device = torch.device("cpu")
            with mock.patch.multiple(
                network_injection,
                **{name: getattr(network_injection, name) for name in assigned},
                OLD_NETWORK="oldset_[esm2_t6_8m]_network.h5",
                NEW_EMBEDDINGS="newset_[esm2_t6_8m]_embeddings.h5",
                EMBED_DIR=embed_dir,
                NETWORK_DIR=network_dir,
                EXECUTION_MODE="auto",
                DEVICE_SELECTION="auto",
                set_start_method=mock.DEFAULT,
                _benchmark_injection_plans=mock.Mock(return_value=[plan]),
                process_cpu_tasks=cpu_tasks,
            ), redirect_stdout(io.StringIO()) as printed, redirect_stderr(io.StringIO()):
                network_injection.run_injection()

        text = printed.getvalue()
        self.assertIn("Leaving out the input network's 1 calibration pairs", text)
        headers = list(vectors)
        similarity = _world_similarity(
            {h: np.asarray([v], np.float32) for h, v in vectors.items()}, headers
        )
        decided = _oracle_cut_rule(old_headers, [0, 0, 2], [1, 2, 3], headers, similarity, {2})
        # The calibration pair C-D (about -0.92) neither sets s_min nor counts in K: A-C does, 2 of 6 pairs are kept.
        self.assertEqual(decided["s_min"], similarity[0, 2])
        self.assertGreater(decided["s_min"], 0.9)
        self.assertIn(f"Kept share of the input network, K: {2 / 6:.6f} (", text)
        self.assertIn(f"s_min: {similarity[0, 2]:.6f} (between 'A' and 'C'", text)
        self.assertIn(f"Cut: {decided['cut']:.6f}, set by {decided['set_by']}.", text)
        # The queue is the rule's: every non-edge pair at or above the cut.
        tasks = [task for call in cpu_tasks.call_args_list for task in call.args[0]]
        self.assertEqual(
            [(task[0], task[1]) for task in tasks],
            [(p // 5, p % 5) for p in _oracle_pending(5, decided["required"], decided["cached"], set())],
        )
        # The calibration edge stays in the network as an edge.
        self.assertIn(2 * 5 + 3, decided["required"])


class InjectionResumeScanTests(BatchResumeScanFixture, unittest.TestCase):
    """scan_existing_batches resumes only batches written for this run.

    Any other batch sends the whole folder to <RESULTS_DIR>_BackUp with a
    report, and no pair is reused.
    """

    MATCHING_ATTRS = {
        "embedding_checksum": "checksum",
        "model_name": "test-model",
        "saving_mode": "float32",
        "gap_penalties": np.array([-2.0, 0.0], np.float32),
        "matmul_precision": "ieee_fp32",
    }

    def scan(self, results_dir):
        stdout = io.StringIO()
        with mock.patch.object(
            network_injection, "RESULTS_DIR", results_dir
        ), redirect_stdout(stdout):
            computed = network_injection.scan_existing_batches(
                2, "checksum", "test-model", "float32", [-2.0, 0.0], "ieee_fp32"
            )
        return computed, stdout.getvalue()

    def test_matching_batch_resumes_its_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = os.path.join(temp_dir, "batches")
            self.write_batch(results_dir)
            computed, _output = self.scan(results_dir)
            self.assertEqual(os.listdir(temp_dir), ["batches"])
            self.assertEqual(os.listdir(results_dir), ["batch_00000.h5"])
        # Pair (0, 1) of a 2-sequence set is key 0 * 2 + 1.
        self.assertEqual(computed, {1})

    def test_each_identity_mismatch_backs_up_the_folder(self):
        cached_gaps = list(np.array([-3.0, 0.0], np.float32))
        cases = {
            "checksum": (
                {"embedding_checksum": "other"},
                "Checksum mismatch in 'batch_00000.h5' ('other' vs current 'checksum')",
            ),
            "model_name": (
                {"model_name": "other-model"},
                "Model name mismatch in 'batch_00000.h5' ('other-model' vs current 'test-model')",
            ),
            "saving_mode": (
                {"saving_mode": "float16"},
                "Saving mode mismatch in 'batch_00000.h5' ('float16' vs current 'float32')",
            ),
            "gap_penalties": (
                {"gap_penalties": np.array([-3.0, 0.0], np.float32)},
                f"Gap penalties mismatch in 'batch_00000.h5' ({cached_gaps} vs current [-2.0, 0.0])",
            ),
            "matmul_precision": (
                {"matmul_precision": "tf32"},
                "Matmul precision mismatch in 'batch_00000.h5' ('tf32' vs 'ieee_fp32')",
            ),
            "missing l_len": (
                {"omit": ("l_len",)},
                "Missing required datasets in batch file 'batch_00000.h5'",
            ),
            "i/j length": (
                {"j": (1, 1)},
                "Dataset length mismatch in batch file 'batch_00000.h5'",
            ),
        }
        for label, (batch, reason) in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as temp_dir:
                results_dir = os.path.join(temp_dir, "batches")
                self.write_batch(results_dir, **batch)
                computed, output = self.scan(results_dir)
                reasons = self.assert_backed_up(results_dir, results_dir + "_BackUp")
                self.assertEqual(computed, set())
                self.assertEqual(reasons, [reason])
                self.assertIn(f"  > Mismatch reason: {reason}\n", output)

    def test_repeat_backup_takes_the_next_free_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = os.path.join(temp_dir, "batches")
            self.write_batch(results_dir, saving_mode="float16")
            self.scan(results_dir)
            self.write_batch(results_dir, saving_mode="bfloat16")
            computed, output = self.scan(results_dir)
            reasons = self.assert_backed_up(results_dir, results_dir + "_BackUp_1")
            self.assertEqual(
                sorted(os.listdir(temp_dir)),
                ["batches", "batches_BackUp", "batches_BackUp_1"],
            )
            first_batch = os.path.join(results_dir + "_BackUp", "batch_00000.h5")
            with h5py.File(first_batch, "r") as hf:
                self.assertEqual(hf.attrs["saving_mode"], "float16")
        self.assertEqual(computed, set())
        self.assertEqual(
            reasons,
            ["Saving mode mismatch in 'batch_00000.h5' ('bfloat16' vs current 'float32')"],
        )
        self.assertIn(
            f"  > Renaming existing batch folder to: '{results_dir}_BackUp_1'\n",
            output,
        )


# ---------------------------------------------------------------------------
# Pair selection (the cut rule) and the final merge against slow, plain
# reference code written here: loops over every pair, no numpy shortcuts for
# the decisions under test.
# ---------------------------------------------------------------------------

def _world_similarity(embeddings, headers):
    """The shared similarity of the new set: max-pooled, length-adjusted cosine (float64, n x n)."""
    lengths = [len(embeddings[h]) for h in headers]
    return Prefilter_Calibration.pair_similarity(embeddings, headers, lengths)


def _oracle_cut_rule(old_headers, old_i, old_j, new_headers, s, calibration):
    """What the cut rule decides, by plain loops. Returns a dict.

    Keys: sparse, cached, required (sets of pair keys i * N + j), K, s_min,
    lowest (first edge with the strictly lowest s, -1 if none), t_prime, cut,
    set_by, kept_after (share of pairs with s >= cut). calibration is the set
    of edge numbers that are recorded calibration edges.
    """
    new_idx = {h: idx for idx, h in enumerate(new_headers)}
    new_n = len(new_headers)
    old_n = len(old_headers)
    total_old = old_n * (old_n - 1) // 2
    sparse = len(old_i) < total_old
    old_set = set(old_headers)
    cached = set()
    for k in range(len(old_i)):
        u_h, v_h = old_headers[old_i[k]], old_headers[old_j[k]]
        if u_h in new_idx and v_h in new_idx:
            u, v = new_idx[u_h], new_idx[v_h]
            cached.add(min(u, v) * new_n + max(u, v))
    pairs = [(i, j) for i in range(new_n) for j in range(i + 1, new_n)]
    info = {"sparse": sparse, "cached": cached, "K": None, "s_min": None, "lowest": -1,
            "t_prime": None, "cut": None, "set_by": None, "kept_after": None}
    if not sparse or len(old_i) == 0:
        # A complete input, or a sparse one with no edge at all (nothing to compare with, as
        # on main): every pair with a new sequence, plus the old edges; no other old-old pair.
        info["required"] = set(cached) | {
            i * new_n + j for i, j in pairs
            if not (new_headers[i] in old_set and new_headers[j] in old_set)
        }
        return info
    info["K"] = (len(old_i) - len(calibration)) / total_old
    for k in range(len(old_i)):
        if k in calibration:
            continue
        u_h, v_h = old_headers[old_i[k]], old_headers[old_j[k]]
        if u_h in new_idx and v_h in new_idx:
            value = float(s[new_idx[u_h], new_idx[v_h]])
            if value == value and (info["lowest"] == -1 or value < info["s_min"]):
                info["s_min"], info["lowest"] = value, k
    values = np.array([s[i, j] for i, j in pairs], dtype=np.float64)
    if values.size == 0:
        info["cut"] = float("inf")
    else:
        info["t_prime"] = float(np.percentile(values, 100.0 * (1.0 - info["K"])))
        if info["lowest"] != -1 and info["s_min"] < info["t_prime"]:
            info["cut"], info["set_by"] = info["s_min"], "s_min"
        else:
            info["cut"], info["set_by"] = info["t_prime"], "t'"
    info["required"] = set(cached) | {i * new_n + j for i, j in pairs if s[i, j] >= info["cut"]}
    info["kept_after"] = (
        sum(1 for i, j in pairs if s[i, j] >= info["cut"]) / len(pairs) if pairs else 0.0
    )
    return info


def _oracle_pending(new_n, required, cached, computed):
    return [
        i * new_n + j
        for i in range(new_n)
        for j in range(i + 1, new_n)
        if (i * new_n + j) in required
        and (i * new_n + j) not in cached
        and (i * new_n + j) not in computed
    ]


def _oracle_merge(new_n, required, cached, new_headers, old_index, old_arrays,
                  actual_idx_map, batches):
    """The original compile_final_output arrays: (i, j, l_score, l_len, g_score, g_len)."""
    old_l_score, old_l_len, old_g_score, old_g_len = old_arrays
    sorted_pairs = sorted(required)
    final = {name: np.zeros(len(sorted_pairs), dtype=dtype) for name, dtype in (
        ("i", np.uint16), ("j", np.uint16), ("l_score", np.float32),
        ("l_len", np.uint16), ("g_score", np.float32), ("g_len", np.uint16))}
    slot = {}
    for idx, pair_id in enumerate(sorted_pairs):
        final["i"][idx] = pair_id // new_n
        final["j"][idx] = pair_id % new_n
        slot[pair_id] = idx
    for pair_id in cached:
        if pair_id in slot:
            idx = slot[pair_id]
            u = old_index[new_headers[pair_id // new_n]]
            v = old_index[new_headers[pair_id % new_n]]
            lo, hi = min(u, v), max(u, v)
            theoretical = lo * len(old_index) - lo * (lo + 1) // 2 + (hi - lo - 1)
            actual = actual_idx_map[theoretical]
            final["l_score"][idx] = old_l_score[actual]
            final["l_len"][idx] = old_l_len[actual]
            final["g_score"][idx] = old_g_score[actual]
            final["g_len"][idx] = old_g_len[actual]
    for arr_i, arr_j, l_score, l_len, g_score, g_len in batches:
        for k in range(len(arr_i)):
            pair_id = min(int(arr_i[k]), int(arr_j[k])) * new_n + max(int(arr_i[k]), int(arr_j[k]))
            if pair_id in slot and pair_id not in cached:
                idx = slot[pair_id]
                final["l_score"][idx] = l_score[k]
                final["l_len"][idx] = l_len[k]
                final["g_score"][idx] = g_score[k]
                final["g_len"][idx] = g_len[k]
    return final


def _random_world(rng, kind, large=False):
    """An old network and a new sequence set that drops some old sequences and adds new ones.

    Each sequence has 1 to 4 residue rows drawn from a tiny grid, so that many
    similarities are exactly equal (ties) and the lengths differ.
    kind: "dense" (a complete input), "sparse_record" (a recorded calibration
    sample), "sparse_plain", "sparse_all_dropped" (every old edge loses an end
    in the new set) or "sparse_no_edge". large: about 300 to 380 new sequences,
    so that pair keys (min * N + max) pass 65,535 and wrap in a 16-bit index array.
    Returns (embeddings, old_headers, new_headers, edges, calibration), edges
    being (i, j) with i < j in old indices.
    """
    pool = [f"s{k}" for k in range(int(rng.integers(380, 420) if large else rng.integers(12, 24)))]
    grid = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 1]], np.float32)
    embeddings = {
        h: grid[rng.integers(len(grid), size=int(rng.integers(1, 5)))] for h in pool
    }
    old_n = int(rng.integers(300, 340) if large else rng.integers(3, 10))
    old_headers = [pool[k] for k in rng.permutation(len(pool))[:old_n]]
    all_pairs = [(i, j) for i in range(old_n) for j in range(i + 1, old_n)]
    if kind == "dense":
        edges = [all_pairs[k] for k in rng.permutation(len(all_pairs))]
    elif kind == "sparse_no_edge":
        edges = []
    elif kind == "sparse_all_dropped" and large:
        # Edges among the first 30 old sequences only, so that most sequences survive the drop.
        few = [(i, j) for (i, j) in all_pairs if j < 30]
        edges = [few[k] for k in rng.permutation(len(few))[: int(rng.integers(1, len(few)))]]
    else:
        count = int(rng.integers(1, len(all_pairs)))
        edges = [all_pairs[k] for k in rng.permutation(len(all_pairs))[:count]]
    calibration = set()
    if kind == "sparse_record":
        calibration = {int(k) for k in rng.permutation(len(edges))[: int(rng.integers(1, len(edges) + 1))]}
    kept = [h for h in old_headers if rng.random() < (0.92 if large else 0.75)]
    if kind == "sparse_all_dropped":
        # Keep only sequences that no edge touches (a dropped one for every edge).
        touched = {old_headers[k] for e in edges for k in e}
        kept = [h for h in old_headers if h not in touched]
    others = [h for h in pool if h not in old_headers]
    added = others[: int(rng.integers(30, 41) if large else rng.integers(0, 7))]
    members = kept + added
    if len(members) < 2:
        members = (members + others[len(added):] + old_headers)[:2] if kind == "sparse_all_dropped" else old_headers[:2]
    new_headers = [members[k] for k in rng.permutation(len(members))]
    return embeddings, old_headers, new_headers, edges, calibration


class NetworkInjectionSelectionTests(unittest.TestCase):
    """The numpy pair selection and merge equal the original per-pair loops."""

    def _glue(self, old_headers, edges, new_headers, random_orientation=None):
        """old_to_new maps and the two ends of every old edge, as run_injection builds them."""
        new_idx = {h: idx for idx, h in enumerate(new_headers)}
        old_to_new = np.array([new_idx.get(h, -1) for h in old_headers], dtype=np.int64)
        old_i = np.array([e[0] for e in edges], dtype=np.int64)
        old_j = np.array([e[1] for e in edges], dtype=np.int64)
        if random_orientation is not None:
            flip = random_orientation.random(len(edges)) < 0.5
            old_i, old_j = np.where(flip, old_j, old_i), np.where(flip, old_i, old_j)
        return old_i, old_j, old_to_new[old_i], old_to_new[old_j]

    KINDS = ("dense", "sparse_record", "sparse_plain", "sparse_all_dropped", "sparse_no_edge")

    def test_cut_rule_helpers_equal_the_oracle(self):
        set_by_seen = {"s_min": 0, "t'": 0}
        seen_kinds = set()
        for seed, large in [(s, False) for s in range(150)] + [(s, True) for s in range(300, 310)]:
            rng = np.random.default_rng(seed)
            kind = self.KINDS[seed % len(self.KINDS)]
            embeddings, old_headers, new_headers, edges, calibration = _random_world(rng, kind, large)
            new_n = len(new_headers)
            old_n = len(old_headers)
            if large:
                self.assertGreater(new_n, 300)  # keys pass 65,535
            similarity = _world_similarity(embeddings, new_headers)
            old_i, old_j, edge_u, edge_v = self._glue(old_headers, edges, new_headers, rng)
            skip = np.zeros(len(edges), dtype=bool)
            skip[list(calibration)] = True
            expected = _oracle_cut_rule(
                old_headers, old_i.tolist(), old_j.tolist(), new_headers, similarity, calibration
            )
            block_cells = int(rng.choice([1, new_n, 2 * new_n + 1, 7, 10 ** 6]))
            chunk = int(rng.choice([977, 1 << 22] if large else [1, 2, 3, 1 << 22]))
            with self.subTest(seed=seed, kind=kind, block_cells=block_cells, chunk=chunk), \
                    mock.patch.object(network_injection, "_PAIR_BLOCK_CELLS", block_cells), \
                    mock.patch.object(network_injection, "_EDGE_CHUNK", chunk), \
                    mock.patch.object(network_injection, "_PENDING_CHUNK", chunk):
                cached = network_injection._cached_pair_keys(edge_u, edge_v, new_n)
                self.assertEqual(cached.dtype, np.int64)
                self.assertEqual(cached.tolist(), sorted(expected["cached"]))
                old_set = set(old_headers)
                is_old = np.array([h in old_set for h in new_headers])
                total_pairs = new_n * (new_n - 1) // 2
                if expected["sparse"] and len(edges) > 0:
                    kept_share = (len(edges) - len(calibration)) / (old_n * (old_n - 1) // 2)
                    cut, s_min, lowest, t_prime, set_by = network_injection._similarity_cut(
                        similarity, edge_u, edge_v, skip, kept_share
                    )
                    self.assertEqual(lowest, expected["lowest"])
                    self.assertEqual(s_min, expected["s_min"])
                    self.assertEqual(t_prime, expected["t_prime"])
                    self.assertEqual(cut, expected["cut"])
                    self.assertEqual(set_by, expected["set_by"])
                    if set_by:
                        set_by_seen[set_by] += 1
                    seen_kinds.add(kind)
                    required = network_injection._required_pair_keys(
                        new_n, is_old, cached, similarity, cut
                    )
                    kept_count = network_injection._pairs_at_or_above(similarity, cut)
                    self.assertEqual(kept_count / total_pairs if total_pairs else 0.0, expected["kept_after"])
                    # The properties of the rule, checked on the helpers' own output.
                    self.assertEqual(cut, min(v for v in (s_min, t_prime) if v is not None))
                    self.assertGreater(kept_count, kept_share * total_pairs - 1)  # >= K up to rounding
                    first_keys = set(required.tolist())
                    self.assertTrue(set(cached.tolist()) <= first_keys)  # no old edge is dropped
                    for key in first_keys - set(cached.tolist()):
                        self.assertGreaterEqual(similarity[key // new_n, key % new_n], cut)
                    pair_rows, pair_columns = np.triu_indices(new_n, 1)
                    at_or_above = pair_rows[similarity[pair_rows, pair_columns] >= cut] * new_n \
                        + pair_columns[similarity[pair_rows, pair_columns] >= cut]
                    self.assertTrue(set(at_or_above.tolist()) <= first_keys)  # every pair at or above the cut
                elif expected["sparse"]:
                    # A sparse input without an edge: every pair with a new sequence, no old-old pair.
                    seen_kinds.add(kind)
                    required = network_injection._required_pair_keys(new_n, is_old, cached, None, None)
                else:
                    seen_kinds.add(kind)
                    required = network_injection._required_pair_keys(new_n, is_old, cached, None, None)
                self.assertEqual(required.dtype, np.int64)
                self.assertEqual(required.tolist(), sorted(expected["required"]))

                computed = {
                    int(k) for k in rng.choice(
                        new_n * new_n, int(rng.integers(0, 10)), replace=True
                    )
                } | set(sorted(expected["required"])[: int(rng.integers(0, 3))])
                headers = [f"h{k}" for k in range(new_n)]
                pending = _oracle_pending(new_n, expected["required"], expected["cached"], computed)
                self.assertEqual(
                    list(network_injection._iter_pending_pairs(
                        headers, expected["required"], expected["cached"], computed
                    )),
                    [(p // new_n, p % new_n, headers[p // new_n], headers[p % new_n]) for p in pending],
                )
                keys = network_injection._pending_keys(
                    required, cached, np.array(sorted(computed), dtype=np.int64), new_n
                )
                self.assertEqual(keys.tolist(), pending)
                np.testing.assert_array_equal(
                    np.bincount(keys // new_n, minlength=new_n),
                    np.bincount([p // new_n for p in pending], minlength=new_n),
                )
                kinds = [0, 0, 0]
                for p in pending:
                    kinds[2 - int(is_old[p // new_n]) - int(is_old[p % new_n])] += 1
                self.assertEqual(list(network_injection._pending_kinds(keys, is_old, new_n)), kinds)
        # Both ways the rule can go, and every kind of input, were exercised.
        self.assertGreater(set_by_seen["s_min"], 5)
        self.assertGreater(set_by_seen["t'"], 5)
        self.assertEqual(seen_kinds, set(self.KINDS))

    def test_share_cut_is_the_percentile_aligns_prefilter_takes(self):
        rng = np.random.default_rng(3)
        for n in (2, 3, 7, 40):
            similarity = rng.random((n, n))
            for share in (0.0, 0.1, 0.5, 0.62, 0.999, 1.0):
                with self.subTest(n=n, share=share):
                    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
                    expected = float(np.percentile(similarity[upper], 100.0 * (1.0 - share)))
                    self.assertEqual(network_injection._share_cut(similarity, share), expected)
        self.assertIsNone(network_injection._share_cut(np.ones((1, 1)), 0.5))

    def test_a_network_align_made_keeps_exactly_its_own_pairs(self):
        # Align's prefilter keeps the pairs at or above the strength percentile; the share of those
        # pairs, fed back into the cut, must select the same pairs again (a no-op injection queues nothing).
        rng = np.random.default_rng(11)
        for n in (5, 12, 60):
            similarity = rng.random((n, n))
            upper = np.triu(np.ones((n, n), dtype=bool), k=1)
            total = n * (n - 1) // 2
            for strength in (10.0, 25.0, 50.0, 75.0, 90.0):
                with self.subTest(n=n, strength=strength):
                    threshold = float(np.percentile(similarity[upper], strength))
                    kept = (similarity >= threshold) & upper
                    share = np.count_nonzero(kept) / total
                    cut = network_injection._share_cut(similarity, share)
                    again = (similarity >= cut) & upper
                    np.testing.assert_array_equal(again, kept)

    def test_lowest_edge_similarity_is_the_first_edge_among_equal_values(self):
        similarity = np.array([[1.0, 0.5, 0.5, 0.9], [0.5, 1.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.7], [0.9, 0.5, 0.7, 1.0]])
        edge_u = np.array([0, 2, 0, 1, 1, 0])
        edge_v = np.array([3, 3, 1, 2, 3, 2])
        skip = np.zeros(6, dtype=bool)
        for chunk in (1, 2, 4, 100):
            with mock.patch.object(network_injection, "_EDGE_CHUNK", chunk):
                self.assertEqual(
                    network_injection._lowest_edge_similarity(similarity, edge_u, edge_v, skip),
                    (0.5, 2),
                )
                skip[2] = True
                self.assertEqual(
                    network_injection._lowest_edge_similarity(similarity, edge_u, edge_v, skip),
                    (0.5, 3),
                )
                skip[2] = False
                # An edge that lost an end (-1) never counts.
                self.assertEqual(
                    network_injection._lowest_edge_similarity(
                        similarity, np.array([0, -1]), np.array([3, 1]), np.zeros(2, bool)
                    ),
                    (0.9, 0),
                )
        # An edge at similarity 1.0 is a lowest edge like any other; a NaN never is.
        ones = np.ones((3, 3))
        self.assertEqual(
            network_injection._lowest_edge_similarity(ones, np.array([0]), np.array([1]), np.zeros(1, bool)),
            (1.0, 0),
        )
        with_nan = np.array([[1.0, np.nan, 0.8], [np.nan, 1.0, 0.7], [0.8, 0.7, 1.0]])
        self.assertEqual(
            network_injection._lowest_edge_similarity(
                with_nan, np.array([0, 0, 1]), np.array([1, 2, 2]), np.zeros(3, bool)
            ),
            (0.7, 2),
        )
        # No counted edge: nothing.
        self.assertEqual(
            network_injection._lowest_edge_similarity(
                similarity, np.array([0]), np.array([1]), np.ones(1, bool)
            ),
            (None, -1),
        )

    def test_the_cut_is_the_lower_of_s_min_and_t_prime_in_both_cases(self):
        # Six sequences, 15 pairs. The off-diagonal similarities are 0.0, 0.1, ... 1.4 / 1.5 (distinct).
        n = 6
        similarity = np.zeros((n, n))
        values = iter(np.arange(15) / 10.0)
        for i in range(n):
            similarity[i, i] = 1.0
            for j in range(i + 1, n):
                similarity[i, j] = similarity[j, i] = next(values)
        # Pair (0, 1) has 0.0 ... pair (4, 5) 1.4.
        edge_u = np.array([2, 3])
        edge_v = np.array([4, 5])
        skip = np.zeros(2, dtype=bool)
        low_edge = similarity[2, 4]
        high_edge = similarity[3, 5]
        # A tiny kept share: t' is high; the lowest edge lies below it, so it sets the cut.
        cut, s_min, lowest, t_prime, set_by = network_injection._similarity_cut(similarity, edge_u, edge_v, skip, 2 / 15)
        self.assertEqual((set_by, cut, s_min, lowest), ("s_min", low_edge, low_edge, 0))
        self.assertGreater(t_prime, low_edge)
        # A large kept share: t' is low, below both edges: it sets the cut and keeps K.
        cut, s_min, lowest, t_prime, set_by = network_injection._similarity_cut(similarity, edge_u, edge_v, skip, 14 / 15)
        self.assertEqual(set_by, "t'")
        self.assertEqual(cut, t_prime)
        self.assertLess(t_prime, low_edge)
        self.assertLessEqual(t_prime, min(low_edge, high_edge))
        # Equal: the lowest edge at exactly t' does not set the cut (t' does), same value.
        exact = float(network_injection._share_cut(similarity, 0.5))
        edge_exact_u, edge_exact_v = np.nonzero(np.triu(similarity == exact, 1))
        cut, s_min, lowest, t_prime, set_by = network_injection._similarity_cut(
            similarity, edge_exact_u, edge_exact_v, np.zeros(edge_exact_u.size, bool), 0.5
        ) if edge_exact_u.size else (exact, exact, 0, exact, "t'")
        self.assertEqual((cut, set_by), (exact, "t'"))
        # No edge in the new set: the cut is t'.
        cut, s_min, lowest, t_prime, set_by = network_injection._similarity_cut(
            similarity, np.array([-1]), np.array([4]), skip[:1], 0.5
        )
        self.assertEqual((set_by, s_min, lowest, cut), ("t'", None, -1, t_prime))
        # A new set with fewer than two sequences has no pair: the cut is infinite.
        cut, _, _, t_prime, set_by = network_injection._similarity_cut(
            np.ones((1, 1)), np.array([0]), np.array([0]), skip[:1], 0.5
        )
        self.assertEqual((cut, t_prime, set_by), (float("inf"), None, None))

    def test_a_new_pair_at_exactly_the_cut_is_required(self):
        # Pair (0, 1) is the lowest old edge; the pair (2, 3) has exactly the same similarity.
        similarity = np.array([
            [1.0, 0.6, 0.9, 0.9],
            [0.6, 1.0, 0.9, 0.9],
            [0.9, 0.9, 1.0, 0.6],
            [0.9, 0.9, 0.6, 1.0],
        ])
        is_old = np.array([True, True, False, False])
        cached = np.array([0 * 4 + 1], dtype=np.int64)
        required = network_injection._required_pair_keys(4, is_old, cached, similarity, 0.6)
        self.assertEqual(required.tolist(), [1, 2, 3, 6, 7, 11])
        # Just above the cut the cached edge stays (never dropped), the pair at 0.6 does not.
        self.assertEqual(
            network_injection._required_pair_keys(4, is_old, cached, similarity, 0.61).tolist(),
            [1, 2, 3, 6, 7],
        )
        # Old-old pairs without an edge are required too when they reach the cut.
        both_old = np.array([True, True, True, True])
        self.assertEqual(
            network_injection._required_pair_keys(4, both_old, cached, similarity, 0.9).tolist(),
            [1, 2, 3, 6, 7],
        )
    def test_pair_keys_stay_exact_far_beyond_32_bits(self):
        # N = 100,000: keys reach 10 ** 10, past both 2 ** 31 and 2 ** 32.
        n = 100_000
        edge_u = np.array([99998, 5, 70000, -1, 3, 5], dtype=np.int64)
        edge_v = np.array([99999, 99990, 3, 7, -1, 99990], dtype=np.int64)
        cached = network_injection._cached_pair_keys(edge_u, edge_v, n)
        expected_cached = sorted({99998 * n + 99999, 5 * n + 99990, 3 * n + 70000})
        self.assertEqual(cached.dtype, np.int64)
        self.assertEqual(cached.tolist(), expected_cached)
        self.assertGreater(int(cached.max()), 2 ** 33)

        extra = [10 * n + 11, 99990 * n + 99991, 40_000 * n + 80_000]
        # A wrong-orientation key and one whose row is past the sequence count never queue.
        invalid = [50 * n + 50, 70_000 * n + 60_000, n * n + 1]
        required = network_injection._as_sorted_keys(set(expected_cached + extra + invalid))
        self.assertEqual(required.tolist(), sorted(expected_cached + extra + invalid))
        computed = network_injection._as_sorted_keys([extra[0], 123])
        pending = network_injection._pending_keys(required, cached, computed, n)
        self.assertEqual(pending.tolist(), sorted(extra[1:]))

        position, found = network_injection._locate(required, np.array(extra + [7, 10 ** 10 + 5], dtype=np.int64))
        self.assertEqual(found.tolist(), [True, True, True, False, False])
        np.testing.assert_array_equal(required[position[:3]], extra)
        self.assertEqual(
            network_injection._members(cached, np.array(expected_cached[::-1] + [1], dtype=np.int64)).tolist(),
            [True, True, True, False],
        )

        headers = [f"h{k}" for k in range(n)]
        queued = list(network_injection._iter_pending_keys(headers, pending, n))
        self.assertEqual(
            queued,
            [(k // n, k % n, headers[k // n], headers[k % n]) for k in sorted(extra[1:])],
        )
        self.assertEqual(
            list(network_injection._iter_pending_pairs(headers, set(required.tolist()), set(cached.tolist()), set(computed.tolist()))),
            queued,
        )

        # Flat condensed indices of an old network of 100,000 sequences pass 4.9 * 10 ** 9.
        flat = np.array([5_000_000_000, 7, 4_999_000_000, 7], dtype=np.int64)
        index = network_injection._OldEdgeIndex(flat)
        self.assertEqual(
            index[np.array([5_000_000_000, 7, 4_999_000_000, 8, 5_000_000_001])].tolist(),
            [0, 3, 2, -1, -1],
        )
        self.assertEqual(network_injection._sorted_unique(np.array([10 ** 10, 3, 10 ** 10, 3], dtype=np.int64)).tolist(), [3, 10 ** 10])

    def _write_batch(self, results_dir, name, pairs, scores, index_dtype=np.uint32):
        arr_i = np.array([p[0] for p in pairs], dtype=index_dtype)
        arr_j = np.array([p[1] for p in pairs], dtype=index_dtype)
        with h5py.File(os.path.join(results_dir, name), "w") as hf:
            hf.create_dataset("i", data=arr_i)
            hf.create_dataset("j", data=arr_j)
            hf.create_dataset("l_score", data=np.array([s[0] for s in scores], np.float32))
            hf.create_dataset("l_len", data=np.array([s[1] for s in scores], np.uint16))
            hf.create_dataset("g_score", data=np.array([s[2] for s in scores], np.float32))
            hf.create_dataset("g_len", data=np.array([s[3] for s in scores], np.uint16))

    def test_compile_equals_the_original_merge_on_random_inputs(self):
        # The large cases have keys past 65,535 and store the batch rows as uint16.
        for seed, large in [(s, False) for s in range(40)] + [(s, True) for s in range(100, 104)]:
            rng = np.random.default_rng(1000 + seed)
            embeddings, old_headers, new_headers, edges, _calibration = _random_world(
                rng, ("dense", "sparse_plain")[seed % 2], large
            )
            batch_index_dtype = np.uint16 if large else np.uint32
            new_n = len(new_headers)
            old_n = len(old_headers)
            old_i = np.array([e[0] for e in edges], dtype=np.int64)
            old_j = np.array([e[1] for e in edges], dtype=np.int64)
            count = len(edges)
            old_arrays = (
                (np.arange(count) + 10.5).astype(np.float32),
                (np.arange(count) + 3).astype(np.uint16),
                (np.arange(count) + 500.25).astype(np.float32),
                (np.arange(count) + 4).astype(np.uint16),
            )
            flat = old_i * old_n - old_i * (old_i + 1) // 2 + (old_j - old_i - 1)
            legacy_map = np.full(old_n * (old_n - 1) // 2, -1, dtype=np.int64)
            legacy_map[flat] = np.arange(count)
            old_index = {h: k for k, h in enumerate(old_headers)}
            decided = _oracle_cut_rule(
                old_headers, old_i.tolist(), old_j.tolist(), new_headers,
                _world_similarity(embeddings, new_headers), set(),
            )
            cached, required = decided["cached"], decided["required"]
            # Batches: pairs to compute, a pair written twice, pairs that are not
            # required, a cached pair with a wrong score, and reversed (j, i) rows.
            everything = [(i, j) for i in range(new_n) for j in range(i + 1, new_n)]
            batch_pairs = [
                [everything[k] for k in rng.permutation(len(everything))[: int(rng.integers(0, len(everything) + 1))]]
                for _ in range(int(rng.integers(1, 4)))
            ]
            batches = []
            with tempfile.TemporaryDirectory() as temp_dir:
                results_dir = os.path.join(temp_dir, "batches")
                os.makedirs(results_dir)
                for number, pairs in enumerate(batch_pairs):
                    pairs = [p[::-1] if rng.random() < 0.3 else p for p in pairs]
                    scores = [(float(100 * number + n + 0.5), n % 50 + 1, float(7 * number + n + 0.25), n % 40 + 2)
                              for n in range(len(pairs))]
                    self._write_batch(results_dir, f"batch_{number:05d}.h5", pairs, scores, batch_index_dtype)
                    batches.append((
                        np.array([p[0] for p in pairs], dtype=np.int64),
                        np.array([p[1] for p in pairs], dtype=np.int64),
                        np.array([s[0] for s in scores], np.float32),
                        np.array([s[1] for s in scores], np.uint16),
                        np.array([s[2] for s in scores], np.float32),
                        np.array([s[3] for s in scores], np.uint16),
                    ))
                expected = _oracle_merge(
                    new_n, required, cached, new_headers, old_index, old_arrays, legacy_map,
                    batches if batch_pairs else [],
                )
                kinds = {
                    "sets and legacy map": (set(required), set(cached), legacy_map),
                    "arrays and edge index": (
                        np.array(sorted(required), dtype=np.int64),
                        np.array(sorted(cached), dtype=np.int64),
                        network_injection._OldEdgeIndex(flat),
                    ),
                }
                for label, (req, cac, index_map) in kinds.items():
                    output = os.path.join(temp_dir, label.split()[0], "net_[model]_network.h5")
                    with self.subTest(seed=seed, inputs=label), mock.patch.object(
                        network_injection, "RESULTS_DIR", results_dir
                    ), mock.patch.object(
                        network_injection, "FINAL_OUTPUT_NET", output
                    ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        network_injection.compile_final_output(
                            new_headers=new_headers, seq_lens=[5] * new_n,
                            required_pairs=req, cached_old_pairs=cac, new_N=new_n,
                            old_header_to_idx=old_index,
                            old_l_score=old_arrays[0], old_l_len=old_arrays[1],
                            old_g_score=old_arrays[2], old_g_len=old_arrays[3],
                            actual_idx_map=index_map, current_checksum="c",
                            model_name="model", saving_mode="float32",
                            gap_penalties=[-2.0, 0.0],
                        )
                        with h5py.File(output, "r") as hf:
                            for name in ("i", "j", "l_score", "l_len", "g_score", "g_len"):
                                np.testing.assert_array_equal(hf[name][:], expected[name], err_msg=name)
                                self.assertEqual(hf[name].dtype, expected[name].dtype)

    def _run_injection(self, temp_dir, rng, kind, batch_size, large=False):
        """Run run_injection on a random world with stubbed alignment; return what it did.

        large: 300 to 380 sequences, so pair keys pass 65,535; the old network's
        i/j are uint16 as always and the left-over batch stores its rows as uint16.
        """
        world, old_headers, new_headers, edges, calibration = _random_world(rng, kind, large)
        embed_dir = os.path.join(temp_dir, "embed")
        network_dir = os.path.join(temp_dir, "network")
        os.makedirs(embed_dir)
        os.makedirs(network_dir)
        embeddings = os.path.join(embed_dir, "newset_[esm2_t6_8m]_embeddings.h5")
        with h5py.File(embeddings, "w") as hf:
            group = Embedding_HDF5.create_metadata_first_file(
                hf, new_headers, ["K" * len(world[h]) for h in new_headers], "esm2_t6_8m", "float32",
            )
            for header in new_headers:
                group.create_dataset(header, data=world[header].astype(np.float32))
            Embedding_HDF5.mark_generation_complete(hf)
        old_n = len(old_headers)
        count = len(edges)
        with h5py.File(os.path.join(network_dir, "oldset_[esm2_t6_8m]_network.h5"), "w") as hf:
            hf.attrs["model_name"] = "esm2_t6_8m"
            hf.attrs["saving_mode"] = "float32"
            hf.attrs["gap_penalties"] = np.asarray([-2.0, 0.0], np.float32)
            hf.attrs["matmul_precision"] = "ieee_fp32"
            hf.create_dataset("headers", data=np.array(old_headers, dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
            hf.create_dataset("seq_lens", data=np.ones(old_n, np.uint16))
            for name, data in (
                ("i", [e[0] for e in edges]), ("j", [e[1] for e in edges]),
                ("l_score", np.arange(count) + 10.5), ("l_len", np.arange(count) + 1),
                ("g_score", np.arange(count) + 500.25), ("g_len", np.arange(count) + 2),
            ):
                dtype = np.float32 if name.endswith("score") else np.uint16
                hf.create_dataset(name, data=np.asarray(data, dtype))
            if calibration:
                ordered = sorted(calibration)
                Prefilter_Calibration.write_record(hf, Prefilter_Calibration.PrefilterRecord(
                    strength=50.0, threshold=0.9, pooling="max", length_power=2.0, kept=count, skipped=count,
                    sample=Prefilter_Calibration.CalibrationSample(
                        np.array([edges[k][0] for k in ordered]),
                        np.array([edges[k][1] for k in ordered]),
                        np.ones(len(ordered)),
                    ),
                ), np.uint16)
        similarity = _world_similarity(world, new_headers)
        old_i = [e[0] for e in edges]
        old_j = [e[1] for e in edges]
        decided = _oracle_cut_rule(old_headers, old_i, old_j, new_headers, similarity, calibration)
        cached, required = decided["cached"], decided["required"]
        sparse = decided["sparse"]
        new_n = len(new_headers)
        number = lambda header: int(header[1:])
        stub = lambda i, j: (
            1000.5 + 10 * number(new_headers[i]) + number(new_headers[j]),
            number(new_headers[i]) + number(new_headers[j]) + 1,
            2000.25 + 7 * number(new_headers[i]) + number(new_headers[j]),
            number(new_headers[i]) + number(new_headers[j]) + 2,
        )
        # A batch left by an earlier run: some pending pairs, a pair no longer
        # required, and a cached pair with a wrong score; all must be handled.
        pending = _oracle_pending(new_n, required, cached, set())
        resumed = {}
        computed_all = set()
        if rng.random() < 0.85 and pending:
            results_dir = os.path.join(network_dir, "newset_[esm2_t6_8m]_network_temp")
            os.makedirs(results_dir)
            chosen = [pending[k] for k in rng.permutation(len(pending))[: int(rng.integers(1, len(pending) + 1))]]
            rows = [(p // new_n, p % new_n) for p in chosen]
            scores = [(7000.5 + n, n + 1, 8000.25 + n, n + 2) for n in range(len(rows))]
            resumed = {p: scores[n] for n, p in enumerate(chosen)}
            extra = [p for p in range(new_n * new_n)
                     if p // new_n < p % new_n and p not in required][:1] + sorted(cached)[:1]
            for p in extra:
                rows.append((p // new_n, p % new_n))
                scores.append((9999.5, 9, 9999.25, 9))
            computed_all = set(resumed) | set(extra)
            self._write_batch(
                results_dir, "batch_00000.h5", rows, scores, np.uint16 if large else np.uint32
            )
            with h5py.File(os.path.join(results_dir, "batch_00000.h5"), "a") as hf:
                hf.attrs["embedding_checksum"] = network_injection.calculate_file_hash(embeddings)
                hf.attrs["model_name"] = "esm2_t6_8m"
                hf.attrs["saving_mode"] = "float32"
                hf.attrs["gap_penalties"] = np.asarray([-2.0, 0.0], np.float32)
                hf.attrs["matmul_precision"] = "ieee_fp32"
        queue = _oracle_pending(new_n, required, cached, set(resumed))

        def compute(tasks, *_args, **_kwargs):
            return [(i, j, *stub(i, j)) for i, j, _h_i, _h_j in tasks]

        cpu_tasks = mock.Mock(side_effect=compute)
        plan = mock.Mock(variant="scalar", lanes=1)
        plan.candidate.is_cpu = True
        plan.candidate.device = torch.device("cpu")
        assigned = (
            "_old_seq_set", "_new_seq_set", "_model_name", "RESULTS_DIR",
            "FINAL_OUTPUT_NET", "CONFIG_FILE", "LOCAL_GAP_P", "GLOBAL_GAP_P",
        )
        printed = io.StringIO()
        with mock.patch.multiple(
            network_injection,
            **{name: getattr(network_injection, name) for name in assigned},
            OLD_NETWORK="oldset_[esm2_t6_8m]_network.h5",
            NEW_EMBEDDINGS="newset_[esm2_t6_8m]_embeddings.h5",
            EMBED_DIR=embed_dir, NETWORK_DIR=network_dir,
            EXECUTION_MODE="auto", DEVICE_SELECTION="auto", BATCH_SIZE=batch_size,
            set_start_method=mock.DEFAULT,
            _benchmark_injection_plans=mock.Mock(return_value=[plan]),
            process_cpu_tasks=cpu_tasks,
        ), redirect_stdout(printed), redirect_stderr(io.StringIO()), mock.patch.object(
            Prefilter_Calibration, "pair_similarity", wraps=Prefilter_Calibration.pair_similarity
        ) as similarity_probe:
            network_injection.run_injection()
        with h5py.File(os.path.join(network_dir, "newset_[esm2_t6_8m]_network.h5"), "r") as hf:
            actual = {name: hf[name][:] for name in ("i", "j", "l_score", "l_len", "g_score", "g_len")}
        expected_rows = []
        old_position = {h: n for n, h in enumerate(old_headers)}
        edge_row = {(min(e), max(e)): n for n, e in enumerate(edges)}
        for p in sorted(required):
            if p in cached:
                u = old_position[new_headers[p // new_n]]
                v = old_position[new_headers[p % new_n]]
                k = edge_row[(min(u, v), max(u, v))]
                row = (k + 10.5, k + 1, k + 500.25, k + 2)
            elif p in resumed:
                row = resumed[p]
            else:
                row = stub(p // new_n, p % new_n)
            expected_rows.append((p // new_n, p % new_n, *row))
        return {
            "tasks": [list(call.args[0]) for call in cpu_tasks.call_args_list],
            "queue": queue, "new_n": new_n, "printed": printed.getvalue(),
            "actual": actual, "expected_rows": expected_rows,
            "decided": decided, "required": required, "cached": cached,
            "resumed": resumed, "computed_all": computed_all,
            "calibration": calibration, "sparse": sparse,
            "old_headers": old_headers, "new_headers": new_headers, "edges": edges,
            "similarity": similarity, "similarity_calls": similarity_probe.call_count,
        }

    def test_run_injection_follows_the_cut_rule_end_to_end(self):
        resumed_seen = 0
        kinds_seen = set()
        for seed, large in [(s, False) for s in range(60)] + [(s, True) for s in range(200, 210)]:
            kind = self.KINDS[seed % len(self.KINDS)]
            rng = np.random.default_rng(5000 + seed)
            batch_size = int(rng.choice([997, 500000] if large else [1, 3, 7, 500000]))
            with self.subTest(seed=seed, kind=kind, batch_size=batch_size, large=large), \
                    tempfile.TemporaryDirectory() as temp_dir:
                result = self._run_injection(temp_dir, rng, kind, batch_size, large)
                new_n = result["new_n"]
                if large:
                    self.assertGreater(new_n, 300)  # keys pass 65,535
                queue = result["queue"]
                flat = [task for batch in result["tasks"] for task in batch]
                self.assertEqual([(t[0], t[1]) for t in flat], [(p // new_n, p % new_n) for p in queue])
                self.assertEqual(
                    [len(batch) for batch in result["tasks"]],
                    [len(queue[k:k + batch_size]) for k in range(0, len(queue), batch_size)],
                )
                rows = list(zip(*(result["actual"][n].tolist() for n in ("i", "j", "l_score", "l_len", "g_score", "g_len"))))
                self.assertEqual(rows, result["expected_rows"])
                text = result["printed"]
                self.assertIn(f"Total Required pairs: {len(result['required'])}\n", text)
                self.assertIn(f"Pre-existing cached pairs: {len(result['cached'])}\n", text)
                # Every pair the left-over batch holds that the network requires counts,
                # including its cached extra row.
                self.assertIn(
                    f"Already computed pairs: {len(result['computed_all'] & result['required'])}\n", text
                )
                self.assertIn(f"Pairs queued for calculation: {len(queue)}\n", text)
                decided = result["decided"]
                # The similarity is computed once for a sparse input with an edge, never otherwise.
                self.assertEqual(
                    result["similarity_calls"],
                    1 if decided["sparse"] and decided["K"] is not None else 0,
                )
                if kind == "sparse_no_edge":
                    # Nothing to compare with: every pair with a new sequence is aligned, as on
                    # main, and no pair of two old sequences.
                    new_old_set = set(result["old_headers"])
                    self.assertEqual(
                        set(queue) | set(result["resumed"]),
                        {
                            p for p in range(new_n * new_n)
                            if p // new_n < p % new_n
                            and not (result["new_headers"][p // new_n] in new_old_set
                                     and result["new_headers"][p % new_n] in new_old_set)
                        },
                    )
                    self.assertIn(
                        "every pair with a new sequence is aligned and no pair of two old sequences.", text
                    )
                if kind == "dense":
                    # A complete input needs no similarity: every pair with a new sequence is aligned.
                    self.assertEqual(
                        set(queue) | set(result["resumed"]),
                        {
                            p for p in range(new_n * new_n)
                            if p // new_n < p % new_n
                            and not (result["new_headers"][p // new_n] in set(result["old_headers"])
                                     and result["new_headers"][p % new_n] in set(result["old_headers"]))
                        },
                    )
                old_set = set(result["old_headers"])
                new_headers = result["new_headers"]
                old_ends = [
                    int(new_headers[p // new_n] in old_set) + int(new_headers[p % new_n] in old_set)
                    for p in queue
                ]
                self.assertIn(
                    f"  > Queued by kind: {old_ends.count(2):,} old-old (both sequences were in the input "
                    f"network, but not joined by an edge), {old_ends.count(1):,} old-new, "
                    f"{old_ends.count(0):,} new-new\n",
                    text,
                )
                if decided["sparse"] and decided["K"] is not None:
                    self.assertIn(f"Kept share of the input network, K: {decided['K']:.6f} (", text)
                    if decided["lowest"] != -1:
                        edge = result["edges"][decided["lowest"]]
                        self.assertIn(
                            f"s_min: {decided['s_min']:.6f} (between '{result['old_headers'][edge[0]]}' "
                            f"and '{result['old_headers'][edge[1]]}', score: {decided['lowest'] + 10.5:.4f})\n",
                            text,
                        )
                    else:
                        self.assertIn("s_min does not apply.\n", text)
                    if decided["set_by"] is None:
                        self.assertIn("fewer than two sequences", text)
                    else:
                        self.assertIn(f"t': {decided['t_prime']:.6f}\n", text)
                        self.assertIn(f"Cut: {decided['cut']:.6f}, set by {decided['set_by']}.", text)
                        self.assertIn(
                            f"Kept share of the new set's pairs at or above the cut: "
                            f"{decided['kept_after']:.6f} (input network: {decided['K']:.6f})\n",
                            text,
                        )
                    # The old scan and its lines are gone.
                    self.assertNotIn("below the threshold", text)
                elif decided["sparse"]:
                    self.assertIn("The input network has no edge", text)
                else:
                    self.assertNotIn("Kept share", text)
                kinds_seen.add(kind)
                resumed_seen += bool(result["resumed"])
        self.assertGreater(resumed_seen, 5)
        self.assertEqual(kinds_seen, set(self.KINDS))


if __name__ == "__main__":
    unittest.main()
