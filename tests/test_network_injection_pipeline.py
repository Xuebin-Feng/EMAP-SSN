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


if __name__ == "__main__":
    unittest.main()
