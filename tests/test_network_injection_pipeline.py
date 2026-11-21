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
        # One-residue embeddings, so each mean is the vector itself. A-B and
        # A-C are kept pairs (cosine about 0.98 and up); C-D is the prefilter's
        # calibration pair, far below its cut (about -0.92). The new sequence
        # E's cosines to A-D lie between the two, so counting the calibration
        # pair as the lowest kept edge would align all four E pairs.
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

            cpu_tasks = mock.Mock(return_value=[])
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
        cosine_ac = 1.0 / np.linalg.norm(vectors["C"])
        self.assertIn(f"Setting cosine similarity prefiltering threshold to: {cosine_ac:.6f}", text)
        tasks = [task for call in cpu_tasks.call_args_list for task in call.args[0]]
        self.assertEqual([task for task in tasks if "E" in task[2:]], [])


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
# Pair selection and the final merge against the original per-pair loops.
# The oracle functions below are the loops the numpy code replaced, kept
# slow and plain on purpose.
# ---------------------------------------------------------------------------

def _unit_cosines(vectors):
    """The tool's all-vs-all cosine of (n, d) vectors: float32, clipped."""
    mean_embs = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(mean_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    norm_embs = mean_embs / norms
    return np.clip(np.dot(norm_embs, norm_embs.T), -1.0, 1.0)


def _oracle_selection(old_headers, old_i, old_j, new_headers, cos_sim, calibration, sparse):
    """(threshold, lowest edge, cached pairs, required pairs) by the original loops."""
    new_idx = {h: idx for idx, h in enumerate(new_headers)}
    new_n = len(new_headers)
    threshold = -1.0
    lowest = -1
    if sparse and len(old_i) > 0:
        min_val = 1.0
        for k in range(len(old_i)):
            if k in calibration:
                continue
            u_h = old_headers[old_i[k]]
            v_h = old_headers[old_j[k]]
            if u_h in new_idx and v_h in new_idx:
                sim = float(cos_sim[new_idx[u_h], new_idx[v_h]])
                if sim < min_val:
                    min_val, lowest = sim, k
        threshold = min_val if lowest != -1 else 0.0
    old_set = set(old_headers)
    cached = set()
    for k in range(len(old_i)):
        u_h = old_headers[old_i[k]]
        v_h = old_headers[old_j[k]]
        if u_h in new_idx and v_h in new_idx:
            u, v = new_idx[u_h], new_idx[v_h]
            cached.add(min(u, v) * new_n + max(u, v))
    required = set()
    for i in range(new_n):
        is_old_i = new_headers[i] in old_set
        for j in range(i + 1, new_n):
            is_old_j = new_headers[j] in old_set
            keep = False
            if is_old_i and is_old_j:
                keep = (i * new_n + j) in cached
            elif not sparse or (cos_sim is not None and cos_sim[i, j] >= threshold):
                keep = True
            if keep:
                required.add(i * new_n + j)
    return threshold, lowest, cached, required


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


def _random_world(rng, kind):
    """An old network and a new sequence set that drops some old sequences and adds new ones.

    Vectors come from a tiny grid so that many cosines are exactly equal (ties).
    kind: "dense", "sparse_record" or "sparse_plain".
    """
    pool = [f"s{k}" for k in range(int(rng.integers(12, 24)))]
    grid = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 1]], np.float32)
    vectors = {h: grid[int(rng.integers(len(grid)))] for h in pool}
    old_n = int(rng.integers(3, 10))
    old_headers = [pool[k] for k in rng.permutation(len(pool))[:old_n]]
    kept = [h for h in old_headers if rng.random() < 0.75]
    others = [h for h in pool if h not in old_headers]
    added = others[: int(rng.integers(0, 7))]
    members = kept + added
    if len(members) < 2:
        members = old_headers[:2]
    new_headers = [members[k] for k in rng.permutation(len(members))]
    all_pairs = [(i, j) for i in range(old_n) for j in range(i + 1, old_n)]
    if kind == "dense":
        edges = [all_pairs[k] for k in rng.permutation(len(all_pairs))]
    else:
        count = int(rng.integers(1, len(all_pairs)))
        edges = [all_pairs[k] for k in rng.permutation(len(all_pairs))[:count]]
    calibration = set()
    if kind == "sparse_record":
        calibration = {int(k) for k in rng.permutation(len(edges))[: int(rng.integers(1, len(edges) + 1))]}
    return vectors, old_headers, new_headers, edges, calibration


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

    def test_selection_helpers_equal_the_original_loops(self):
        for seed in range(80):
            rng = np.random.default_rng(seed)
            kind = ("dense", "sparse_record", "sparse_plain")[seed % 3]
            vectors, old_headers, new_headers, edges, calibration = _random_world(rng, kind)
            cos_sim = _unit_cosines([vectors[h] for h in new_headers])
            new_n = len(new_headers)
            sparse = kind != "dense"
            old_i, old_j, edge_u, edge_v = self._glue(old_headers, edges, new_headers, rng)
            skip = np.zeros(len(edges), dtype=bool)
            skip[list(calibration)] = True
            expected = _oracle_selection(
                old_headers, old_i.tolist(), old_j.tolist(), new_headers, cos_sim,
                calibration, sparse,
            )
            block_cells = int(rng.choice([1, new_n, 2 * new_n + 1, 7, 10 ** 6]))
            chunk = int(rng.choice([1, 2, 3, 1 << 22]))
            with self.subTest(seed=seed, kind=kind, block_cells=block_cells, chunk=chunk), \
                    mock.patch.object(network_injection, "_PAIR_BLOCK_CELLS", block_cells), \
                    mock.patch.object(network_injection, "_EDGE_CHUNK", chunk), \
                    mock.patch.object(network_injection, "_PENDING_CHUNK", chunk):
                threshold = -1.0
                if sparse:
                    value, lowest = network_injection._lowest_edge_cosine(
                        cos_sim, edge_u, edge_v, skip
                    )
                    threshold = value if lowest != -1 else 0.0
                    self.assertEqual(lowest, expected[1])
                    self.assertEqual(threshold, expected[0])
                    # The scan for edges below the cut: counts what the loop counted.
                    for limit in (threshold - 1e-5, threshold + 0.2, 2.0):
                        count, first = network_injection._edges_below_cosine(
                            cos_sim, edge_u, edge_v, skip, limit
                        )
                        loop = [
                            (k, cos_sim[edge_u[k], edge_v[k]]) for k in range(len(edges))
                            if k not in calibration and edge_u[k] >= 0 and edge_v[k] >= 0
                            and cos_sim[edge_u[k], edge_v[k]] < limit
                        ]
                        self.assertEqual(count, len(loop))
                        self.assertEqual([k for k, _ in first], [k for k, _ in loop[:10]])
                cached = network_injection._cached_pair_keys(edge_u, edge_v, new_n)
                self.assertEqual(cached.dtype, np.int64)
                self.assertEqual(cached.tolist(), sorted(expected[2]))
                old_set = set(old_headers)
                is_old = np.array([h in old_set for h in new_headers])
                required = network_injection._required_pair_keys(
                    new_n, is_old, cached, cos_sim if sparse else None, threshold
                )
                self.assertEqual(required.dtype, np.int64)
                self.assertEqual(required.tolist(), sorted(expected[3]))

                computed = {
                    int(k) for k in rng.choice(
                        new_n * new_n, int(rng.integers(0, 10)), replace=True
                    )
                } | set(sorted(expected[3])[: int(rng.integers(0, 3))])
                headers = [f"h{k}" for k in range(new_n)]
                pending = _oracle_pending(new_n, expected[3], expected[2], computed)
                self.assertEqual(
                    list(network_injection._iter_pending_pairs(
                        headers, expected[3], expected[2], computed
                    )),
                    [(p // new_n, p % new_n, headers[p // new_n], headers[p % new_n]) for p in pending],
                )
                keys = network_injection._pending_keys(
                    required, cached, np.array(sorted(computed), dtype=np.int64), new_n
                )
                self.assertEqual(keys.tolist(), pending)
                counts = np.bincount(keys // new_n, minlength=new_n)
                for row in range(new_n):
                    self.assertEqual(
                        counts[row], sum(1 for p in pending if p // new_n == row)
                    )

    def test_lowest_cosine_is_the_first_edge_among_equal_values(self):
        cos_sim = np.array([[1.0, 0.5, 0.5, 0.9], [0.5, 1.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.7], [0.9, 0.5, 0.7, 1.0]], np.float32)
        edge_u = np.array([0, 2, 0, 1, 1, 0])
        edge_v = np.array([3, 3, 1, 2, 3, 2])
        skip = np.zeros(6, dtype=bool)
        for chunk in (1, 2, 4, 100):
            with mock.patch.object(network_injection, "_EDGE_CHUNK", chunk):
                self.assertEqual(
                    network_injection._lowest_edge_cosine(cos_sim, edge_u, edge_v, skip),
                    (0.5, 2),
                )
                skip[2] = True
                self.assertEqual(
                    network_injection._lowest_edge_cosine(cos_sim, edge_u, edge_v, skip),
                    (0.5, 3),
                )
                skip[2] = False
        # Nothing below 1.0: no edge, and the caller falls back to 0.0.
        ones = np.ones((2, 2), np.float32)
        self.assertEqual(
            network_injection._lowest_edge_cosine(ones, np.array([0]), np.array([1]), np.zeros(1, bool)),
            (1.0, -1),
        )

    def test_a_new_pair_at_exactly_the_cut_is_required(self):
        # A-B is the lowest old edge; the new pair C-D has the same cosine.
        cos_sim = np.array([
            [1.0, 0.6, 0.9, 0.9],
            [0.6, 1.0, 0.9, 0.9],
            [0.9, 0.9, 1.0, 0.6],
            [0.9, 0.9, 0.6, 1.0],
        ], np.float32)
        is_old = np.array([True, True, False, False])
        cached = np.array([0 * 4 + 1], dtype=np.int64)
        required = network_injection._required_pair_keys(4, is_old, cached, cos_sim, float(cos_sim[0, 1]))
        self.assertEqual(required.tolist(), [1, 2, 3, 6, 7, 11])
        self.assertEqual(
            network_injection._required_pair_keys(4, is_old, cached, cos_sim, 0.61).tolist(),
            [1, 2, 3, 6, 7],
        )

    def _write_batch(self, results_dir, name, pairs, scores):
        arr_i = np.array([p[0] for p in pairs], dtype=np.uint32)
        arr_j = np.array([p[1] for p in pairs], dtype=np.uint32)
        with h5py.File(os.path.join(results_dir, name), "w") as hf:
            hf.create_dataset("i", data=arr_i)
            hf.create_dataset("j", data=arr_j)
            hf.create_dataset("l_score", data=np.array([s[0] for s in scores], np.float32))
            hf.create_dataset("l_len", data=np.array([s[1] for s in scores], np.uint16))
            hf.create_dataset("g_score", data=np.array([s[2] for s in scores], np.float32))
            hf.create_dataset("g_len", data=np.array([s[3] for s in scores], np.uint16))

    def test_compile_equals_the_original_merge_on_random_inputs(self):
        for seed in range(40):
            rng = np.random.default_rng(1000 + seed)
            vectors, old_headers, new_headers, edges, _calibration = _random_world(
                rng, ("dense", "sparse_plain")[seed % 2]
            )
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
            _t, _k, cached, required = _oracle_selection(
                old_headers, old_i.tolist(), old_j.tolist(), new_headers,
                _unit_cosines([vectors[h] for h in new_headers]), set(), False,
            )
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
                    self._write_batch(results_dir, f"batch_{number:05d}.h5", pairs, scores)
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

    def _run_injection(self, temp_dir, rng, kind, batch_size):
        """Run run_injection on a random world with stubbed alignment; return what it did."""
        vectors, old_headers, new_headers, edges, calibration = _random_world(rng, kind)
        embed_dir = os.path.join(temp_dir, "embed")
        network_dir = os.path.join(temp_dir, "network")
        os.makedirs(embed_dir)
        os.makedirs(network_dir)
        embeddings = os.path.join(embed_dir, "newset_[esm2_t6_8m]_embeddings.h5")
        with h5py.File(embeddings, "w") as hf:
            group = Embedding_HDF5.create_metadata_first_file(
                hf, new_headers, ["K"] * len(new_headers), "esm2_t6_8m", "float32",
            )
            for header in new_headers:
                group.create_dataset(header, data=vectors[header][None, :].astype(np.float32))
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
        cos_sim = _unit_cosines([vectors[h] for h in new_headers])
        sparse = kind != "dense"
        old_i = [e[0] for e in edges]
        old_j = [e[1] for e in edges]
        threshold, _lowest, cached, required = _oracle_selection(
            old_headers, old_i, old_j, new_headers, cos_sim, calibration, sparse
        )
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
            self._write_batch(results_dir, "batch_00000.h5", rows, scores)
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
        ), redirect_stdout(printed), redirect_stderr(io.StringIO()):
            network_injection.run_injection()
        with h5py.File(os.path.join(network_dir, "newset_[esm2_t6_8m]_network.h5"), "r") as hf:
            actual = {name: hf[name][:] for name in ("i", "j", "l_score", "l_len", "g_score", "g_len")}
        expected_rows = []
        for p in sorted(required):
            if p in cached:
                u = old_headers.index(new_headers[p // new_n])
                v = old_headers.index(new_headers[p % new_n])
                k = [n for n in range(count) if {edges[n][0], edges[n][1]} == {u, v}][-1]
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
            "threshold": threshold, "required": required, "cached": cached,
            "resumed": resumed, "computed_all": computed_all,
            "calibration": calibration, "sparse": sparse,
        }

    def test_run_injection_equals_the_original_selection_end_to_end(self):
        resumed_seen = 0
        for seed in range(36):
            kind = ("dense", "sparse_record", "sparse_plain")[seed % 3]
            rng = np.random.default_rng(5000 + seed)
            batch_size = int(rng.choice([1, 3, 7, 500000]))
            with self.subTest(seed=seed, kind=kind, batch_size=batch_size), \
                    tempfile.TemporaryDirectory() as temp_dir:
                result = self._run_injection(temp_dir, rng, kind, batch_size)
                new_n = result["new_n"]
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
                if result["sparse"] and "Old network is empty" not in text:
                    self.assertIn(
                        f"Setting cosine similarity prefiltering threshold to: {result['threshold']:.6f}\n", text
                    )
                resumed_seen += bool(result["resumed"])
        self.assertGreater(resumed_seen, 5)


if __name__ == "__main__":
    unittest.main()
