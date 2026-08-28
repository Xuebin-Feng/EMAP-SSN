import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
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

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Embedding_Injection as embedding_injection
    import Embedding_HDF5
    import Network_Injection as network_injection


class EmbeddingInjectionPluginTests(unittest.TestCase):
    EXPECTED_PLUGINS = {
        "ankh_base": "ankh",
        "ankh_large": "ankh",
        "esm2_t6_8m": "esm2",
        "esm2_t12_35m": "esm2",
        "esm2_t30_150m": "esm2",
        "esm2_t33_650m": "esm2",
        "esm2_t36_3b": "esm2",
        "esm2_t48_15b": "esm2",
        "esmc_300m": "esmc",
        "esmc_600m": "esmc",
        "esmc_6b": "esmc_6b_api",
        "prost_t5": "prost_t5",
        "prot_bert": "prot_bert",
    }

    def test_every_declared_model_resolves_to_the_expected_plugin(self):
        for model_name, expected_module in self.EXPECTED_PLUGINS.items():
            with self.subTest(model_name=model_name):
                plugin = embedding_injection.find_model_plugin(model_name)

                self.assertIsNotNone(plugin)
                self.assertEqual(plugin.__name__, expected_module)
                self.assertTrue(callable(plugin.load_model))
                self.assertTrue(callable(plugin.get_embedding))

    def test_esmc_6b_selects_remote_api_plugin(self):
        plugin = embedding_injection.find_model_plugin("esmc_6b")

        self.assertEqual(plugin.__name__, "esmc_6b_api")
        self.assertIn("API_MODEL_MAPPINGS", vars(plugin))

    def test_load_model_delegates_to_selected_plugin(self):
        plugin = mock.Mock()
        plugin.__name__ = "test_plugin"
        plugin.load_model.return_value = mock.sentinel.model
        plugin.get_embedding = mock.Mock()

        with mock.patch.object(
            embedding_injection,
            "find_model_plugin",
            return_value=plugin,
        ), mock.patch.object(
            embedding_injection.Hardware_Utils,
            "get_optimal_device",
            return_value=mock.sentinel.device,
        ):
            model_obj, device, selected_plugin = embedding_injection.load_model(
                "test_model"
            )

        plugin.load_model.assert_called_once_with(
            "test_model",
            mock.sentinel.device,
        )
        self.assertIs(model_obj, mock.sentinel.model)
        self.assertIs(device, mock.sentinel.device)
        self.assertIs(selected_plugin, plugin)

    def test_get_embedding_delegates_sequence_processing_to_plugin(self):
        expected = np.ones((3, 5), dtype=np.float16)
        plugin = mock.Mock()
        plugin.get_embedding.return_value = expected

        actual = embedding_injection.get_embedding(
            "AB-CD",
            mock.sentinel.model,
            mock.sentinel.device,
            plugin,
            np.float16,
        )

        plugin.get_embedding.assert_called_once_with(
            "AB-CD",
            mock.sentinel.model,
            mock.sentinel.device,
            np.float16,
        )
        self.assertIs(actual, expected)

    def test_unknown_model_reports_unsupported_plugin(self):
        with mock.patch.object(
            embedding_injection,
            "find_model_plugin",
            return_value=None,
        ), self.assertRaisesRegex(
            ValueError,
            "not supported by any available plugin",
        ):
            embedding_injection.load_model("not_a_model")


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

    def test_injection_benchmark_runs_each_setup_once_with_internal_warmup(self):
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
            timer = kwargs["benchmark_timer"]
            timer.start()
            timer.stop()
            return []

        with mock.patch.object(
            network_injection, "DEVICE_SELECTION", "auto"
        ), mock.patch.object(
            network_injection, "ACCELERATOR_TUNE_PAIRS", 2
        ), mock.patch.object(
            network_injection, "ACCELERATOR_CONFIRM_PAIRS", 3
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
                warmup_task_count=3,
            )

        self.assertTrue(plans)
        self.assertEqual(execute.call_count, 5)
        self.assertEqual([len(call.args[1]) for call in execute.call_args_list], [4, 6, 6, 6, 6])
        self.assertEqual(
            [call.kwargs["warmup_task_count"] for call in execute.call_args_list],
            [2, 3, 3, 3, 3],
        )

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
            with mock.patch.object(
                network_injection,
                "NEW_EMBEDDINGS",
                h5_path,
            ), mock.patch.object(
                network_injection,
                "calculate_file_hash",
            ) as calculate_hash, redirect_stdout(output):
                network_injection.run_injection()

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

            def imap_unordered(self, function, tasks, chunksize):
                for task in tasks:
                    events.append(f"pair-{task[1]}")
                    yield function(task)

        class Timer:
            def start(self):
                events.append("timer-start")

            def stop(self):
                events.append("timer-stop")

        tasks = [(0, column, "a", f"h{column}") for column in range(1, 5)]
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
                warmup_task_count=2,
                benchmark_timer=Timer(),
            )

        self.assertEqual(FakePool.instances, 1)
        self.assertEqual(len(results), 4)
        self.assertEqual(
            events,
            [
                "pool-open",
                "pair-1",
                "pair-2",
                "timer-start",
                "pair-3",
                "pair-4",
                "timer-stop",
                "pool-close",
            ],
        )

    def test_process_batch_never_writes_paths_dataset(self):
        expected = [(0, 1, 1.0, 1, 2.0, 1)]
        tasks = [(0, 1, "a", "b")]

        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(
                    network_injection,
                    "RESULTS_DIR",
                    temp_dir,
                ), mock.patch.object(
                    network_injection.Hardware_Utils,
                    "get_optimal_device",
                    return_value=torch.device("cuda"),
                ), mock.patch.object(
                    network_injection,
                    "process_accelerated_tasks",
                    return_value=expected,
                ), mock.patch.object(
                    network_injection,
                    "process_cpu_tasks",
                ):
            network_injection.process_batch(
                tasks,
                batch_id=5,
                workers=4,
                new_emb_path="unused.h5",
                embedding_checksum="checksum",
                model_name="test-model",
                saving_mode="float32",
                gap_penalties=[-2.0, 0.0],
            )

            output_path = os.path.join(temp_dir, "batch_00005.h5")
            with h5py.File(output_path, "r") as hf:
                self.assertNotIn("paths", hf)
                self.assertEqual(hf.attrs["embedding_checksum"], "checksum")
                self.assertEqual(hf.attrs["matmul_precision"], "ieee_fp32")

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


if __name__ == "__main__":
    unittest.main()
