"""Tests of Align_Similarity_Matrix's tiled and GPU batch execution paths:
execution-mode filtering, batched/tiled score matrices, CUDA/XPU OOM failover,
MPS unified-memory window skipping, tiled streaming batches, memory release,
allocator caps, and multi-lane CUDA synchronization.
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
UTILITIES_DIR = os.path.join(SRC_DIR, "utilities")
TOOLS_DIR = os.path.join(SRC_DIR, "tools")
for _folder in (SRC_DIR, UTILITIES_DIR, TOOLS_DIR):
    if _folder not in sys.path:
        sys.path.insert(0, _folder)

# The tests package points tool imports at a missing settings file.
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as similarity_matrix
    import Embedding_Alignment_Engine as alignment_engine

from tests.align_pipeline_fixtures import (  # noqa: E402
    FakeStreamBackend,
    small_tile_plan,
    write_random_embeddings,
)
from tests.embedding_fixtures import write_embedding_arrays  # noqa: E402


class TiledAlignmentExecutionTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(similarity_matrix, "std_mean_support_cache", {})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_execution_mode_filters_candidate_variants(self):
        cpu = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        cuda = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cuda:0", "CUDA", torch.device("cuda:0"), "cuda"
        )
        xpu = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "xpu:0", "XPU", torch.device("xpu:0"), "xpu"
        )
        mps = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "mps", "MPS", torch.device("mps"), "mps"
        )
        # Candidates in the order cpu, cuda, xpu, mps.
        expectations = {
            "auto": (
                ["scalar"],
                ["scalar", "tiled"],
                ["scalar", "tiled"],
                ["scalar", "tiled"],
            ),
            "scalar": (
                ["scalar"], ["scalar"], ["scalar"], ["scalar"]
            ),
            "tiled": ([], ["tiled"], ["tiled"], ["tiled"]),
        }
        for mode, expected_variants in expectations.items():
            with self.subTest(mode=mode), mock.patch.object(
                similarity_matrix, "EXECUTION_MODE", mode
            ), mock.patch.object(
                similarity_matrix,
                "tiled_accelerator_support",
                side_effect=lambda device, require_memory=False: (
                    device.type in {"cuda", "xpu", "mps"}, "mock capability"
                ),
            ):
                for candidate, expected in zip(
                    (cpu, cuda, xpu, mps), expected_variants
                ):
                    self.assertEqual(
                        similarity_matrix._execution_variants(candidate),
                        expected,
                    )

    def test_forced_tiled_mode_rejects_missing_compatible_accelerator(self):
        cpu = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        with mock.patch.object(
            similarity_matrix, "EXECUTION_MODE", "tiled"
        ), mock.patch.object(
            similarity_matrix.Hardware_Utils,
            "get_available_devices",
            return_value=[cpu],
        ), self.assertRaisesRegex(ValueError, "CUDA/ROCm or XPU"):
            similarity_matrix._benchmark_processing_plans(
                [(0, 1, "a", "b")],
                workers=1,
                input_h5="unused.h5",
                batch_id=0,
            )

    def test_forced_tiled_mode_accepts_xpu_capability(self):
        xpu = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "xpu:0", "XPU", torch.device("xpu:0"), "xpu"
        )
        with mock.patch.object(
            similarity_matrix, "EXECUTION_MODE", "tiled"
        ), mock.patch.object(
            similarity_matrix, "DEVICE_SELECTION", "xpu:0"
        ), mock.patch.object(
            similarity_matrix.Hardware_Utils,
            "get_available_devices",
            return_value=[xpu],
        ), mock.patch.object(
            similarity_matrix,
            "tiled_accelerator_support",
            return_value=(True, "mock XPU support"),
        ):
            self.assertEqual(
                similarity_matrix._validate_execution_mode_hardware(),
                "tiled",
            )

    def test_run_job_distributor_skips_windowed_session_for_mps_unified_memory(self):
        # Apple Silicon MPS shares unified RAM with the host; Align_Similarity_Matrix
        # must skip WindowedPairSession for MPS (both auto and manual selection)
        # while keeping MPS eligible for tiled/scalar execution via EmbeddingTileStore.
        rng = np.random.default_rng(20261125)
        n = 12
        headers = [f"s{i:02d}" for i in range(n)]
        arrays = [rng.standard_normal((80, 64)).astype(np.float16) for _ in range(n)]
        total_bytes = sum(a.nbytes for a in arrays)
        cpu_cand = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        mps_cand = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "mps", "Apple MPS", torch.device("mps"), "mps"
        )
        cuda_cand = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cuda:0", "CUDA", torch.device("cuda:0"), "cuda"
        )
        mps_tiled_plan = similarity_matrix.Hardware_Utils.BenchmarkResult(
            mps_cand, 250.0, lanes=1, variant="tiled"
        )

        saved_globals = (
            "MODEL_NAME", "SEQUENCE_SET", "FULL_INPUT_HDF5",
            "RESULTS_DIR", "FINAL_OUTPUT_NET",
            "active_matmul_precision", "active_embedding_store",
            "active_sequence_lengths",
        )

        with tempfile.TemporaryDirectory() as tmp:
            embed_dir = os.path.join(tmp, "embed")
            os.makedirs(embed_dir)
            h5_name = "mps_skip_[esm2_t6_8m]_embeddings.h5"
            h5_path = os.path.join(embed_dir, h5_name)
            write_embedding_arrays(
                h5_path,
                dict(zip(headers, arrays)),
                model_name="esm2_t6_8m",
                saving_mode="float16",
            )

            for device_sel, exec_mode in (
                ("auto", "auto"),
                ("mps", "auto"),
                ("mps", "tiled"),
            ):
                with self.subTest(device_selection=device_sel, execution_mode=exec_mode):
                    net_dir = os.path.join(tmp, f"net_{device_sel}_{exec_mode}")
                    os.makedirs(net_dir, exist_ok=True)
                    out_buf = io.StringIO()
                    plan_window = mock.Mock(
                        side_effect=AssertionError("windowing must not be planned for MPS")
                    )
                    tiled_support = mock.Mock(return_value=(True, "mock MPS"))
                    bench = mock.Mock(return_value=[mps_tiled_plan])
                    with mock.patch.multiple(
                        similarity_matrix,
                        **{k: getattr(similarity_matrix, k) for k in saved_globals},
                        INPUT_HDF5=h5_name,
                        EMBED_DIR=embed_dir,
                        NETWORK_DIR=net_dir,
                        EDGE_PREFILTERING=False,
                        PREFILTER_STRENGTH=0,
                        BATCH_SIZE=20,
                        WORKERS=2,
                        HOST_CACHE_GB="auto",
                        DEVICE_SELECTION=device_sel,
                        EXECUTION_MODE=exec_mode,
                        ACCELERATOR_PRECISION="ieee_fp32",
                        set_start_method=mock.DEFAULT,
                        tiled_accelerator_support=tiled_support,
                        plan_windowed_budgets=plan_window,
                        _benchmark_processing_plans=bench,
                        _run_batch_with_ranked_plans=mock.Mock(return_value=0),
                        compile_final_output=mock.Mock(),
                    ), mock.patch.object(
                        similarity_matrix.Hardware_Utils,
                        "get_available_devices",
                        return_value=[cpu_cand, mps_cand],
                    ), mock.patch.object(
                        alignment_engine,
                        "resolve_host_cache_bytes",
                        return_value=int(total_bytes * 0.4),
                    ), redirect_stdout(out_buf), redirect_stderr(io.StringIO()):
                        rc = similarity_matrix.run_job_distributor()
                        store_after = similarity_matrix.active_embedding_store

                    self.assertIsNone(rc)
                    self.assertFalse(store_after.window_active)
                    plan_window.assert_not_called()
                    bench.assert_called_once()
                    self.assertNotIn("Activating partial-RAM sliding window", out_buf.getvalue())
                    if exec_mode == "auto":
                        # c.backend != "mps" short-circuits before probing tiled support.
                        tiled_support.assert_not_called()

            # When a non-MPS tiled accelerator also exists in auto mode, window_device
            # skips MPS and passes the discrete accelerator device to plan_windowed_budgets.
            net_dir = os.path.join(tmp, "net_auto_cuda_skip_mps")
            os.makedirs(net_dir, exist_ok=True)
            plan_window_cuda = mock.Mock(return_value=None)
            with mock.patch.multiple(
                similarity_matrix,
                **{k: getattr(similarity_matrix, k) for k in saved_globals},
                INPUT_HDF5=h5_name,
                EMBED_DIR=embed_dir,
                NETWORK_DIR=net_dir,
                EDGE_PREFILTERING=False,
                PREFILTER_STRENGTH=0,
                BATCH_SIZE=20,
                WORKERS=2,
                HOST_CACHE_GB="auto",
                DEVICE_SELECTION="auto",
                EXECUTION_MODE="auto",
                ACCELERATOR_PRECISION="ieee_fp32",
                set_start_method=mock.DEFAULT,
                tiled_accelerator_support=mock.Mock(return_value=(True, "mock")),
                plan_windowed_budgets=plan_window_cuda,
                _benchmark_processing_plans=mock.Mock(return_value=[mps_tiled_plan]),
                _run_batch_with_ranked_plans=mock.Mock(return_value=0),
                compile_final_output=mock.Mock(),
            ), mock.patch.object(
                similarity_matrix.Hardware_Utils,
                "get_available_devices",
                return_value=[cpu_cand, mps_cand, cuda_cand],
            ), mock.patch.object(
                alignment_engine,
                "resolve_host_cache_bytes",
                return_value=int(total_bytes * 0.4),
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = similarity_matrix.run_job_distributor()

            self.assertIsNone(rc)
            plan_window_cuda.assert_called_once()
            self.assertEqual(plan_window_cuda.call_args.kwargs["device"], torch.device("cuda:0"))

    def test_batched_score_matrices_match_scalar_float32_path(self):
        rng = np.random.default_rng(20260821)
        row = rng.normal(size=(7, 12)).astype(np.float32)
        targets = [
            rng.normal(size=(length, 12)).astype(np.float32)
            for length in (3, 5, 8)
        ]
        row_tensor = torch.nn.functional.normalize(torch.as_tensor(row), dim=-1)
        target_tensors = [
            torch.nn.functional.normalize(torch.as_tensor(target), dim=-1)
            for target in targets
        ]

        actual = alignment_engine._batched_score_matrices(
            row_tensor,
            target_tensors,
            [len(target) for target in targets],
        ).numpy()

        for index, target in enumerate(targets):
            expected = similarity_matrix.compute_score_matrix_torch(
                row,
                target,
                torch.device("cpu"),
            )
            np.testing.assert_allclose(
                actual[index, :, :len(target)],
                expected,
                rtol=1e-5,
                atol=1e-5,
            )
            expected_alignment = similarity_matrix.calculate_alignment_data(
                (0, index + 1, expected)
            )
            actual_alignment = similarity_matrix.calculate_alignment_data(
                (0, index + 1, actual[index, :, :len(target)])
            )
            self.assertEqual(actual_alignment[3], expected_alignment[3])
            self.assertEqual(actual_alignment[5], expected_alignment[5])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_tiled_cuda_pipeline_matches_scalar_pair_set(self):
        rng = np.random.default_rng(20260822)
        headers = [f"sequence_{index}" for index in range(6)]
        lengths = [5, 6, 7, 8, 9, 10]
        arrays = {
            header: rng.normal(size=(length, 32)).astype(np.float32)
            for header, length in zip(headers, lengths)
        }
        tasks = [
            (row, column, headers[row], headers[column])
            for row in range(len(headers))
            for column in range(row + 1, len(headers))
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = os.path.join(temp_dir, "embeddings.h5")
            write_embedding_arrays(h5_path, arrays, saving_mode="float32")
            store = alignment_engine.EmbeddingTileStore(h5_path, headers, 0)
            scalar_cuda = similarity_matrix._run_accelerated_pipeline(
                tasks,
                workers=2,
                input_h5=h5_path,
                device=torch.device("cuda:0"),
                batch_id=0,
                accelerator_workers=2,
                show_progress=False,
                matmul_precision="float32",
            )
            scalar_cuda_tf32 = similarity_matrix._run_accelerated_pipeline(
                tasks,
                workers=2,
                input_h5=h5_path,
                device=torch.device("cuda:0"),
                batch_id=0,
                accelerator_workers=2,
                show_progress=False,
                matmul_precision="tf32",
            )
            adaptive_plans = alignment_engine.build_adaptive_tile_plans(
                tasks,
                store=store,
                lengths=lengths,
                device=torch.device("cuda:0"),
                lane_candidates=[2],
            )
            self.assertTrue(adaptive_plans)
            selected_memory_plan = adaptive_plans[0].memory_plan
            tiled = alignment_engine.run_tiled_cuda_pipeline(
                tasks,
                store=store,
                lengths=lengths,
                device=torch.device("cuda:0"),
                workers=2,
                lanes=2,
                alignment_callback=similarity_matrix.calculate_alignment_data,
                precision="float32",
                memory_plan_override=selected_memory_plan,
            )
            tf32 = alignment_engine.run_tiled_cuda_pipeline(
                tasks,
                store=store,
                lengths=lengths,
                device=torch.device("cuda:0"),
                workers=2,
                lanes=2,
                alignment_callback=similarity_matrix.calculate_alignment_data,
                precision="tf32",
                memory_plan_override=selected_memory_plan,
            )

        scalar = []
        for row, column, header_i, header_j in tasks:
            matrix = similarity_matrix.compute_score_matrix_torch(
                arrays[header_i], arrays[header_j], torch.device("cpu")
            )
            scalar.append(
                similarity_matrix.calculate_alignment_data(
                    (row, column, matrix)
                )
            )

        tiled_by_pair = {result[:2]: result for result in tiled}
        scalar_by_pair = {result[:2]: result for result in scalar}
        self.assertEqual(tiled_by_pair.keys(), scalar_by_pair.keys())
        for pair, expected in scalar_by_pair.items():
            actual = tiled_by_pair[pair]
            self.assertEqual((actual[3], actual[5]), (expected[3], expected[5]))
            np.testing.assert_allclose(
                [actual[2], actual[4]],
                [expected[2], expected[4]],
                rtol=1e-5,
                atol=1e-3,
            )
        self.assertTrue(
            alignment_engine.compare_precision_results(tiled, tf32)[0]
        )
        self.assertTrue(
            alignment_engine.compare_precision_results(
                scalar_cuda,
                scalar_cuda_tf32,
            )[0]
        )

    def test_tiled_cuda_oom_retries_then_uses_scalar_without_committed_duplicates(self):
        tasks = [(0, 1, "a", "b")]
        expected = [(0, 1, 1.0, 1, 2.0, 1)]
        sink = mock.Mock()
        sink.checkpoint.return_value = 0
        memory_plan = mock.Mock(matrix_bytes=256 * 1024 * 1024)

        # Only the hardware gate is simulated, as in the XPU twins below; the
        # real CUDA backend still classifies the OOM and empties the cache.
        with mock.patch.object(
            similarity_matrix,
            "tiled_accelerator_support",
            return_value=(True, "mock CUDA support"),
        ), mock.patch.object(
            similarity_matrix,
            "run_tiled_accelerator_pipeline",
            side_effect=torch.cuda.OutOfMemoryError("simulated"),
        ) as tiled, mock.patch.object(
            similarity_matrix,
            "cuda_memory_plan",
            return_value=memory_plan,
        ), mock.patch.object(
            similarity_matrix,
            "_run_accelerated_pipeline",
            return_value=expected,
        ) as scalar, mock.patch.object(
            similarity_matrix,
            "tqdm",
            return_value=mock.Mock(),
        ), redirect_stdout(io.StringIO()):
            actual = similarity_matrix.process_accelerated_tasks(
                tasks,
                workers=2,
                input_h5="unused.h5",
                device=torch.device("cuda:0"),
                batch_id=1,
                accelerator_workers=1,
                embedding_store=mock.Mock(),
                sequence_lengths=[2, 2],
                execution_variant="tiled",
                result_callback=sink,
            )

        self.assertEqual(actual, expected)
        self.assertEqual(tiled.call_count, 4)
        self.assertEqual(sink.rollback.call_count, 4)
        scalar.assert_called_once()
        self.assertIs(scalar.call_args.kwargs["result_callback"], sink)

    def test_selected_adaptive_plan_oom_is_returned_to_ranked_fallback(self):
        tasks = [(0, 1, "a", "b")]
        sink = mock.Mock()
        sink.checkpoint.return_value = 0
        memory_plan = mock.Mock(matrix_bytes=256 * 1024 * 1024)

        with mock.patch.object(
            similarity_matrix,
            "tiled_accelerator_support",
            return_value=(True, "mock CUDA support"),
        ), mock.patch.object(
            similarity_matrix,
            "run_tiled_accelerator_pipeline",
            side_effect=torch.cuda.OutOfMemoryError("simulated"),
        ) as tiled, mock.patch.object(
            similarity_matrix,
            "_run_accelerated_pipeline",
        ) as scalar, mock.patch.object(
            similarity_matrix,
            "tqdm",
            return_value=mock.Mock(),
        ), self.assertRaisesRegex(RuntimeError, "plan remained out of memory"):
            similarity_matrix.process_accelerated_tasks(
                tasks,
                workers=2,
                input_h5="unused.h5",
                device=torch.device("cuda:0"),
                batch_id=1,
                accelerator_workers=1,
                embedding_store=mock.Mock(),
                sequence_lengths=[2, 2],
                execution_variant="tiled",
                result_callback=sink,
                memory_plan_override=memory_plan,
            )

        self.assertEqual(tiled.call_count, 4)
        self.assertEqual(sink.rollback.call_count, 4)
        scalar.assert_not_called()

    def test_ranked_batch_runner_tries_the_next_confirmed_plan(self):
        cuda = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cuda:0", "CUDA", torch.device("cuda:0"), "cuda"
        )
        cpu = similarity_matrix.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        ranked = [
            similarity_matrix.Hardware_Utils.BenchmarkResult(
                cuda, 100.0, lanes=2, variant="tiled"
            ),
            similarity_matrix.Hardware_Utils.BenchmarkResult(
                cpu, 10.0, lanes=1, variant="scalar"
            ),
        ]
        with mock.patch.object(
            similarity_matrix, "DEVICE_SELECTION", "auto"
        ), mock.patch.object(
            similarity_matrix,
            "process_batch",
            side_effect=[RuntimeError("simulated OOM"), None],
        ) as process, redirect_stdout(io.StringIO()):
            active = similarity_matrix._run_batch_with_ranked_plans(
                ranked, 0, [(0, 1, "a", "b")], 0, 1, "unused.h5", "checksum"
            )

        self.assertEqual(active, 1)
        self.assertEqual(process.call_count, 2)
        self.assertEqual(
            process.call_args_list[0].kwargs["device"], torch.device("cuda:0")
        )
        self.assertEqual(
            process.call_args_list[1].kwargs["device"], torch.device("cpu")
        )

    def test_tiled_xpu_uses_backend_neutral_runner_and_batch_callback(self):
        tasks = [(0, 1, "a", "b")]
        expected = [(0, 1, 1.0, 1, 2.0, 1)]
        sink = mock.Mock()
        backend = mock.Mock(device_type="xpu")

        with mock.patch.object(
            similarity_matrix,
            "tiled_accelerator_support",
            return_value=(True, "mock XPU support"),
        ), mock.patch.object(
            similarity_matrix,
            "get_accelerator_backend",
            return_value=backend,
        ), mock.patch.object(
            similarity_matrix,
            "run_tiled_accelerator_pipeline",
            return_value=expected,
        ) as tiled, mock.patch.object(
            similarity_matrix,
            "_run_accelerated_pipeline",
        ) as scalar, mock.patch.object(
            similarity_matrix,
            "tqdm",
            return_value=mock.Mock(),
        ):
            actual = similarity_matrix.process_accelerated_tasks(
                tasks,
                workers=2,
                input_h5="unused.h5",
                device=torch.device("xpu:0"),
                batch_id=7,
                accelerator_workers=2,
                embedding_store=mock.Mock(),
                sequence_lengths=[2, 2],
                execution_variant="tiled",
                result_callback=sink,
            )

        self.assertEqual(actual, expected)
        scalar.assert_not_called()
        tiled.assert_called_once()
        self.assertEqual(tiled.call_args.kwargs["device"], torch.device("xpu:0"))
        self.assertEqual(tiled.call_args.kwargs["lanes"], 2)
        self.assertIs(tiled.call_args.kwargs["result_callback"], sink)

    def test_tiled_xpu_oom_retries_then_falls_back_to_scalar_xpu(self):
        tasks = [(0, 1, "a", "b")]
        expected = [(0, 1, 1.0, 1, 2.0, 1)]
        backend = mock.Mock(device_type="xpu")
        backend.is_out_of_memory.return_value = True
        memory_plan = mock.Mock(matrix_bytes=256 * 1024 * 1024)

        with mock.patch.object(
            similarity_matrix,
            "tiled_accelerator_support",
            return_value=(True, "mock XPU support"),
        ), mock.patch.object(
            similarity_matrix,
            "get_accelerator_backend",
            return_value=backend,
        ), mock.patch.object(
            similarity_matrix,
            "run_tiled_accelerator_pipeline",
            side_effect=RuntimeError("XPU out of memory"),
        ) as tiled, mock.patch.object(
            similarity_matrix,
            "cuda_memory_plan",
            return_value=memory_plan,
        ), mock.patch.object(
            similarity_matrix,
            "_run_accelerated_pipeline",
            return_value=expected,
        ) as scalar, mock.patch.object(
            similarity_matrix,
            "tqdm",
            return_value=mock.Mock(),
        ), redirect_stdout(io.StringIO()):
            actual = similarity_matrix.process_accelerated_tasks(
                tasks,
                workers=2,
                input_h5="unused.h5",
                device=torch.device("xpu:0"),
                batch_id=8,
                accelerator_workers=1,
                embedding_store=mock.Mock(),
                sequence_lengths=[2, 2],
                execution_variant="tiled",
            )

        self.assertEqual(actual, expected)
        self.assertEqual(tiled.call_count, 4)
        self.assertEqual(backend.empty_cache.call_count, 4)
        scalar.assert_called_once()
        self.assertEqual(
            scalar.call_args.args[3], torch.device("xpu:0")
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_tiled_process_batch_streams_and_publishes_complete_cuda_results(self):
        rng = np.random.default_rng(20260823)
        headers = ["a", "b", "c", "d"]
        lengths = [5, 6, 7, 8]
        tasks = [
            (row, column, headers[row], headers[column])
            for row in range(4)
            for column in range(row + 1, 4)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            input_h5 = os.path.join(temp_dir, "embeddings.h5")
            results_dir = os.path.join(temp_dir, "batches")
            os.makedirs(results_dir)
            write_embedding_arrays(
                input_h5,
                {
                    header: rng.normal(size=(length, 16)).astype(np.float32)
                    for header, length in zip(headers, lengths)
                },
                saving_mode="float32",
            )
            store = alignment_engine.EmbeddingTileStore(input_h5, headers, 0)
            with mock.patch.object(
                similarity_matrix, "RESULTS_DIR", results_dir
            ), mock.patch.object(
                store, "get", wraps=store.get
            ) as embedding_reads:
                similarity_matrix.process_batch(
                    tasks,
                    batch_id=0,
                    workers=2,
                    input_h5=input_h5,
                    embedding_checksum="checksum",
                    model_name="model",
                    gap_penalties=[-2.0, 0.0],
                    device=torch.device("cuda:0"),
                    accelerator_workers=2,
                    execution_variant="tiled",
                    matmul_precision="ieee_fp32",
                    embedding_store=store,
                    sequence_lengths=lengths,
                )
            self.assertEqual(embedding_reads.call_count, len(headers))
            output = os.path.join(results_dir, "batch_00000.h5")
            self.assertTrue(os.path.exists(output))
            self.assertFalse(os.path.exists(output + ".partial"))
            with h5py.File(output, "r") as hf:
                self.assertEqual(len(hf["i"]), len(tasks))
                self.assertEqual(hf.attrs["matmul_precision"], "ieee_fp32")


class TiledProducerTests(unittest.TestCase):
    LENGTHS = [5, 3, 8, 2, 6, 7, 4, 9, 1, 5]
    FEATURES = 6

    def run_pipeline(
        self, path, headers, device, lanes, batch, log=None, matrix_budget=None
    ):
        store = alignment_engine.EmbeddingTileStore(path, headers, 0)
        tasks = [
            (row, column, headers[row], headers[column])
            for row in range(len(headers))
            for column in range(row + 1, len(headers))
        ]
        loads = []
        original_load = store.load_indices

        def counted_load(*args):
            loads.append(1)
            return original_load(*args)

        def per_pair(args):
            if log is not None:
                log.append("align")
            return similarity_matrix.calculate_alignment_data(args)

        with mock.patch.object(similarity_matrix, "GLOBAL_GAP_P", 0.0), \
                mock.patch.object(similarity_matrix, "LOCAL_GAP_P", -2.0), \
                mock.patch.object(store, "load_indices", side_effect=counted_load):
            results = alignment_engine.run_tiled_accelerator_pipeline(
                tasks,
                store=store,
                lengths=self.LENGTHS,
                device=device,
                workers=2,
                lanes=lanes,
                alignment_callback=per_pair,
                batch_alignment_callback=(
                    similarity_matrix.calculate_alignment_batch if batch else None
                ),
                memory_plan_override=small_tile_plan(self.FEATURES, lanes),
                matrix_budget_override=matrix_budget,
            )
        return sorted(results), len(loads)

    def test_gathered_targets_match_per_target_padding(self):
        rng = np.random.default_rng(20261007)
        lengths = [4, 9, 6, 1, 7]
        embeddings = [
            torch.nn.functional.normalize(
                torch.as_tensor(rng.normal(size=(length, 5)).astype(np.float32)),
                dim=-1,
            )
            for length in lengths
        ]
        starts = [int(value) for value in np.cumsum([0] + lengths[:-1])]
        pad_row = sum(lengths)
        packed = torch.zeros((pad_row + 1, 5))
        for start, tensor in zip(starts, embeddings):
            packed[start:start + len(tensor)] = tensor
        chosen = [1, 3, 4, 2]
        chosen_lengths = [lengths[index] for index in chosen]

        gathered = alignment_engine._gather_padded_targets(
            packed, [starts[index] for index in chosen], chosen_lengths, pad_row
        )
        expected = torch.zeros((len(chosen), max(chosen_lengths), 5))
        for slot, index in enumerate(chosen):
            expected[slot, :lengths[index]] = embeddings[index]
        self.assertTrue(torch.equal(gathered, expected))

        legacy = alignment_engine._batched_score_matrices(
            embeddings[0], [embeddings[index] for index in chosen], chosen_lengths
        )
        from_gather = alignment_engine._batched_score_matrices(
            embeddings[0], gathered, chosen_lengths
        )
        self.assertTrue(
            torch.equal(legacy.view(torch.int32), from_gather.view(torch.int32))
        )

    def test_batch_callback_matches_per_pair_callback_across_tiles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            headers = write_random_embeddings(
                path, self.LENGTHS, self.FEATURES, seed=11
            )
            with mock.patch.object(
                alignment_engine,
                "get_accelerator_backend",
                return_value=FakeStreamBackend([]),
            ):
                per_pair, per_pair_tiles = self.run_pipeline(
                    path, headers, torch.device("cpu"), lanes=2, batch=False
                )
                batched, batched_tiles = self.run_pipeline(
                    path, headers, torch.device("cpu"), lanes=2, batch=True
                )

        self.assertEqual(len(per_pair), 45)
        self.assertGreaterEqual(per_pair_tiles, 3)
        self.assertEqual(batched_tiles, per_pair_tiles)
        self.assertEqual(batched, per_pair)

    def test_each_pair_matches_direct_scoring_of_its_embeddings(self):
        # A one-byte budget puts every target in its own microbatch, so each
        # result must equal scoring that pair's two embeddings directly.
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            headers = write_random_embeddings(
                path, self.LENGTHS, self.FEATURES, seed=19
            )
            with mock.patch.object(
                alignment_engine,
                "get_accelerator_backend",
                return_value=FakeStreamBackend([]),
            ):
                results, tiles = self.run_pipeline(
                    path, headers, torch.device("cpu"), lanes=2, batch=True,
                    matrix_budget=1,
                )
            with h5py.File(path, "r") as hf:
                embeddings = [
                    alignment_engine._to_normalized_accelerator(
                        hf["embeddings"][header][:], torch.device("cpu"), "float32"
                    )
                    for header in headers
                ]

        self.assertGreaterEqual(tiles, 3)
        expected = []
        with mock.patch.object(similarity_matrix, "GLOBAL_GAP_P", 0.0), \
                mock.patch.object(similarity_matrix, "LOCAL_GAP_P", -2.0):
            for row, column, *_scores in results:
                matrix = alignment_engine._batched_score_matrices(
                    embeddings[row], [embeddings[column]], [self.LENGTHS[column]]
                ).numpy()[0]
                expected.append(
                    similarity_matrix.calculate_alignment_data((row, column, matrix))
                )
        self.assertEqual(len(results), 45)
        self.assertEqual(results, sorted(expected))

    def test_cached_memory_is_released_after_the_last_alignment(self):
        # Callers re-check free memory with mem_get_info before every batch;
        # blocks left in PyTorch's cache would read as used memory there.
        log = []
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            headers = write_random_embeddings(
                path, self.LENGTHS, self.FEATURES, seed=13
            )
            with mock.patch.object(
                alignment_engine,
                "get_accelerator_backend",
                return_value=FakeStreamBackend(log),
            ):
                self.run_pipeline(
                    path, headers, torch.device("cpu"), lanes=1, batch=False,
                    log=log,
                )

        self.assertEqual(log.count("align"), 45)
        self.assertEqual(log[-1], "empty-cache")

    def test_cached_memory_is_released_when_a_batch_fails(self):
        # A failed batch must not leave the cache reserved either: auto mode
        # then retries on the next plan, whose support check reads free memory.
        log = []

        def failing_alignment(args):
            raise RuntimeError("alignment failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            headers = write_random_embeddings(
                path, self.LENGTHS, self.FEATURES, seed=23
            )
            store = alignment_engine.EmbeddingTileStore(path, headers, 0)
            tasks = [(0, column, headers[0], headers[column]) for column in (1, 2)]
            with mock.patch.object(
                alignment_engine,
                "get_accelerator_backend",
                return_value=FakeStreamBackend(log),
            ), self.assertRaisesRegex(RuntimeError, "alignment failed"):
                alignment_engine.run_tiled_accelerator_pipeline(
                    tasks,
                    store=store,
                    lengths=self.LENGTHS,
                    device=torch.device("cpu"),
                    workers=1,
                    lanes=1,
                    alignment_callback=failing_alignment,
                    memory_plan_override=small_tile_plan(self.FEATURES, 1),
                )

        self.assertEqual(log, ["empty-cache"])

    def test_the_allocator_cap_covers_the_run_and_is_lifted_after_a_failure_and_after_success(self):
        for fail in (True, False):
            with self.subTest(fail=fail):
                log = []
                fractions = []
                backend = FakeStreamBackend(log)
                backend.device = torch.device("cpu")  # only a token handed back to the function
                backend.module = SimpleNamespace(
                    set_per_process_memory_fraction=lambda fraction, device: fractions.append(
                        (fraction, device, list(log))
                    )
                )

                def alignment(args):
                    if fail:
                        raise RuntimeError("alignment failed")
                    return (args[0], args[1], 1.0, 1, 1.0, 1)

                with tempfile.TemporaryDirectory() as temp_dir:
                    path = os.path.join(temp_dir, "embeddings.h5")
                    headers = write_random_embeddings(path, self.LENGTHS, self.FEATURES, seed=23)
                    store = alignment_engine.EmbeddingTileStore(path, headers, 0)
                    tasks = [(0, column, headers[0], headers[column]) for column in (1, 2)]
                    plan = small_tile_plan(self.FEATURES, 1)
                    with mock.patch.object(alignment_engine, "_allocator_cap_enabled", True), \
                            mock.patch.object(alignment_engine, "get_accelerator_backend", return_value=backend):
                        run = lambda: alignment_engine.run_tiled_accelerator_pipeline(  # noqa: E731
                            tasks, store=store, lengths=self.LENGTHS, device=torch.device("cpu"), workers=1,
                            lanes=1, alignment_callback=alignment, memory_plan_override=plan,
                        )
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "alignment failed"):
                                run()
                        else:
                            run()

                cap = alignment_engine.allocator_cap_bytes(plan)
                self.assertEqual([call[0] for call in fractions], [cap / plan.total_bytes, 1.0])
                self.assertEqual(fractions[0][2], [])  # set before the run touched the device
                self.assertEqual(fractions[1][2], ["empty-cache"])  # lifted after the cache was handed back

    def test_a_failed_run_lets_go_of_what_its_frames_held_while_the_caller_keeps_the_error(self):
        import traceback
        import weakref

        class Sentinel:
            pass

        references = []

        def failing_alignment(args):
            held = Sentinel()  # stands in for a device tensor in a frame of the failed run
            references.append(weakref.ref(held))
            raise RuntimeError("alignment failed")

        log = []
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            headers = write_random_embeddings(path, self.LENGTHS, self.FEATURES, seed=23)
            store = alignment_engine.EmbeddingTileStore(path, headers, 0)
            tasks = [(0, column, headers[0], headers[column]) for column in (1, 2)]
            caught = None
            with mock.patch.object(
                alignment_engine, "get_accelerator_backend", return_value=FakeStreamBackend(log)
            ):
                try:
                    alignment_engine.run_tiled_accelerator_pipeline(
                        tasks, store=store, lengths=self.LENGTHS, device=torch.device("cpu"), workers=1,
                        lanes=1, alignment_callback=failing_alignment,
                        memory_plan_override=small_tile_plan(self.FEATURES, 1),
                    )
                except RuntimeError as error:
                    caught = error  # kept, as process_batch keeps its last out-of-memory error
        self.assertIsNotNone(caught)
        self.assertTrue(references)
        self.assertTrue(all(reference() is None for reference in references))  # gone while the error is held
        self.assertIs(type(caught), RuntimeError)
        self.assertEqual(str(caught), "alignment failed")
        lines = "".join(traceback.format_exception(caught))
        for name in ("failing_alignment", "collect_cpu", "run_tiled_accelerator_pipeline"):
            self.assertIn(name, lines)
        self.assertEqual(log, ["empty-cache"])

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
    def test_a_failed_cuda_run_leaves_the_devices_allocated_bytes_where_they_were_with_the_error_held(self):
        device = torch.device("cuda:0")
        if torch.cuda.mem_get_info(device)[0] < 2 * (1 << 30):
            self.skipTest("less than 2 GiB of the device is free")
        calls = []

        def alignment(args):
            if calls and calls[0] == "fail":
                raise RuntimeError("alignment failed")
            return (args[0], args[1], 1.0, 1, 1.0, 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            headers = write_random_embeddings(path, self.LENGTHS, self.FEATURES, seed=31)
            store = alignment_engine.EmbeddingTileStore(path, headers, 0)
            tasks = [(0, column, headers[0], headers[column]) for column in range(1, 6)]

            def run():
                return alignment_engine.run_tiled_accelerator_pipeline(
                    tasks, store=store, lengths=self.LENGTHS, device=device, workers=1, lanes=1,
                    alignment_callback=alignment, memory_plan_override=small_tile_plan(self.FEATURES, 1),
                )

            def allocated():
                torch.cuda.synchronize(device)
                return torch.cuda.memory_allocated(device)

            run()  # a first run, so what the device library keeps for good is already allocated
            before_success = allocated()
            run()  # a successful run: whatever it leaves is what a run leaves anyway (cuBLAS keeps a workspace per stream)
            kept_by_a_success = allocated() - before_success
            before = allocated()
            calls.append("fail")
            caught = None
            try:
                run()
            except RuntimeError as error:
                caught = error  # held while the bytes are counted
            kept_by_a_failure = allocated() - before
        self.assertIsNotNone(caught)
        self.assertLessEqual(kept_by_a_failure, kept_by_a_success)  # the failed run's tensors are not among it

    @unittest.skipUnless(
        torch.cuda.is_available() and alignment_engine.is_nvidia_cuda(),
        "requires an NVIDIA CUDA device",
    )
    def test_cuda_lanes_and_tile_reuse_do_not_change_results(self):
        # Several lanes read the reused packed tile while the next tile is
        # queued; any missing stream ordering would change the scores.
        device = torch.device("cuda:0")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            headers = write_random_embeddings(
                path, self.LENGTHS, self.FEATURES, seed=17
            )
            single_lane, tiles = self.run_pipeline(
                path, headers, device, lanes=1, batch=False
            )
            many_lanes, _tiles = self.run_pipeline(
                path, headers, device, lanes=4, batch=True
            )
        self.assertGreaterEqual(tiles, 3)
        self.assertEqual(many_lanes, single_lane)

    @unittest.skipUnless(
        torch.cuda.is_available() and alignment_engine.is_nvidia_cuda(),
        "requires an NVIDIA CUDA device",
    )
    def test_cuda_lanes_wait_for_another_lanes_row_copy(self):
        # Each row is copied on the lane of its first microbatch; its other
        # microbatches run on other lanes. Hold every copy behind a long
        # kernel so those lanes would read the row before it is written
        # unless they wait for the copy.
        copy_row = alignment_engine._copy_tile_row

        def delayed_copy(packed, start, length):
            torch.cuda._sleep(100_000_000)
            return copy_row(packed, start, length)

        device = torch.device("cuda:0")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            headers = write_random_embeddings(
                path, self.LENGTHS, self.FEATURES, seed=29
            )
            expected, _tiles = self.run_pipeline(
                path, headers, device, lanes=1, batch=False
            )
            with mock.patch.object(
                alignment_engine, "_copy_tile_row", side_effect=delayed_copy
            ):
                delayed, _tiles = self.run_pipeline(
                    path, headers, device, lanes=4, batch=True
                )
        self.assertEqual(delayed, expected)


if __name__ == "__main__":
    unittest.main()
