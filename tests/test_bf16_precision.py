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
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
TOOLS_ROOT = os.path.join(SRC_ROOT, "tools")
for path in (SRC_ROOT, TOOLS_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as align
    import Embedding_SSEARCH as ssearch
    import Network_Injection as injection
    from utilities import Embedding_Alignment_Engine as engine
    from utilities.Alignment_Network_HDF5 import _normalized_precision


class Bf16PrecisionTests(unittest.TestCase):
    def test_precision_configuration_aliases_are_canonical(self):
        expected = {
            "auto": "automatic_32bit",
            "automatic_32bit": "automatic_32bit",
            "bf16": "bf16",
            "bfloat16": "bf16",
            "fp32": "float32",
            "float32": "float32",
            "tf32": "tf32",
            "tfloat32": "tf32",
        }
        for supplied, canonical in expected.items():
            with self.subTest(supplied=supplied):
                self.assertEqual(
                    engine.normalize_precision_setting(supplied), canonical
                )
        with self.assertRaises(ValueError):
            engine.normalize_precision_setting("float16")

    def test_tool_defaults_export_automatic_32bit(self):
        self.assertEqual(align.ACCELERATOR_PRECISION, "automatic_32bit")
        self.assertEqual(ssearch.ACCELERATOR_PRECISION, "automatic_32bit")
        self.assertEqual(
            align._allowed_result_precisions("auto"),
            {"ieee_fp32", "tf32"},
        )
        self.assertEqual(
            align._allowed_result_precisions("bf16"), {"bf16"}
        )

    def test_cpu_is_rejected_by_bf16_capability_contract(self):
        engine.bf16_support_cache.clear()
        supported, reason = engine.bf16_accelerator_support(
            torch.device("cpu")
        )
        self.assertFalse(supported)
        self.assertIn("CUDA/ROCm, XPU, or MPS", reason)

    def test_xpu_and_mps_capability_probes_use_backend_sync_and_cache(self):
        class Backend:
            def __init__(self):
                self.synchronize_calls = 0
                self.empty_cache_calls = 0

            def synchronize(self):
                self.synchronize_calls += 1

            def empty_cache(self):
                self.empty_cache_calls += 1

        real_arange = torch.arange

        def cpu_arange(*args, **kwargs):
            kwargs.pop("device", None)
            return real_arange(*args, **kwargs)

        for device_name in ("xpu:0", "mps"):
            with self.subTest(device=device_name):
                backend = Backend()
                engine.bf16_support_cache.clear()
                with mock.patch.object(
                    engine, "get_accelerator_backend", return_value=backend
                ), mock.patch.object(
                    engine.torch, "arange", side_effect=cpu_arange
                ) as arange:
                    first = engine.bf16_accelerator_support(
                        torch.device(device_name), refresh=True
                    )
                    second = engine.bf16_accelerator_support(
                        torch.device(device_name)
                    )
                self.assertTrue(first[0])
                self.assertEqual(second, first)
                self.assertEqual(backend.synchronize_calls, 1)
                self.assertEqual(backend.empty_cache_calls, 1)
                self.assertEqual(arange.call_count, 2)

    def test_bf16_capability_probe_reports_runtime_failure(self):
        class Backend:
            def synchronize(self):
                return None

            def empty_cache(self):
                return None

        real_arange = torch.arange

        def cpu_arange(*args, **kwargs):
            kwargs.pop("device", None)
            return real_arange(*args, **kwargs)

        engine.bf16_support_cache.clear()
        with mock.patch.object(
            engine, "get_accelerator_backend", return_value=Backend()
        ), mock.patch.object(
            engine.torch, "arange", side_effect=cpu_arange
        ), mock.patch.object(
            engine.torch, "mm", side_effect=RuntimeError("unsupported")
        ):
            supported, reason = engine.bf16_accelerator_support(
                torch.device("mps"), refresh=True
            )
        self.assertFalse(supported)
        self.assertIn("unsupported", reason)

    def test_bf16_score_matrix_uses_bf16_mm_and_returns_fp32(self):
        left = np.arange(24, dtype=np.float32).reshape(4, 6) + 1
        right = np.arange(30, dtype=np.float32).reshape(5, 6) + 1
        real_mm = torch.mm
        operand_dtypes = []

        def record_mm(a, b):
            operand_dtypes.append((a.dtype, b.dtype))
            return real_mm(a, b)

        with mock.patch.object(torch, "mm", side_effect=record_mm):
            score = engine.compute_score_matrix_torch(
                left, right, torch.device("cpu"), precision="bf16"
            )
        self.assertEqual(
            operand_dtypes, [(torch.bfloat16, torch.bfloat16)]
        )
        self.assertEqual(score.dtype, np.float32)
        self.assertTrue(np.isfinite(score).all())

    def test_batched_bf16_matmul_returns_fp32_workspace(self):
        row = torch.ones((3, 8), dtype=torch.bfloat16)
        targets = [
            torch.ones((2, 8), dtype=torch.bfloat16),
            torch.ones((4, 8), dtype=torch.bfloat16),
        ]
        result = engine._batched_score_matrices(row, targets, [2, 4])
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(tuple(result.shape), (2, 3, 4))
        self.assertTrue(torch.isfinite(result).all().item())

    def test_align_scalar_bf16_operands_return_fp32_scores(self):
        left = np.arange(24, dtype=np.float32).reshape(4, 6) + 1
        right = np.arange(30, dtype=np.float32).reshape(5, 6) + 1
        real_mm = torch.mm
        operand_dtypes = []

        def record_mm(a, b):
            operand_dtypes.append((a.dtype, b.dtype))
            return real_mm(a, b)

        with mock.patch.object(torch, "mm", side_effect=record_mm):
            score = align.compute_score_matrix_torch(
                left, right, torch.device("cpu"), precision="bf16"
            )
        self.assertEqual(
            operand_dtypes, [(torch.bfloat16, torch.bfloat16)]
        )
        self.assertEqual(score.dtype, np.float32)
        self.assertTrue(np.isfinite(score).all())

    def test_bf16_fixed_query_estimate_uses_two_byte_residency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            with h5py.File(path, "w") as hf:
                group = hf.create_group("embeddings")
                group.create_dataset("a", data=np.ones((3, 8), np.float32))
                group.create_dataset("b", data=np.ones((5, 8), np.float32))
            store = engine.EmbeddingTileStore(path, ["a", "b"], 0)
            kwargs = dict(
                tasks=[(0,), (1,)],
                query_embedding=np.ones((4, 8), np.float32),
                store=store,
                lengths=[3, 5],
                device=torch.device("cuda:0"),
                lanes=1,
                memory_info=(8 << 30, 10 << 30),
            )
            fp32 = engine.estimate_fixed_query_cuda_working_set(
                **kwargs, precision="float32"
            )
            bf16 = engine.estimate_fixed_query_cuda_working_set(
                **kwargs, precision="bf16"
            )
        self.assertLess(bf16.tile_bytes, fp32.tile_bytes)
        self.assertGreater(bf16.transient_bytes, 0)

    def test_bf16_tolerance_is_one_hundredth_per_aligned_residue(self):
        baseline = [(0, 1, 10.0, 10, 20.0, 20)]
        within = [(0, 1, 10.099, 10, 20.199, 20)]
        outside = [(0, 1, 10.101, 10, 20.201, 20)]
        self.assertTrue(
            engine.compare_precision_results(
                baseline,
                within,
                per_residue_tolerance=0.01,
                candidate_label="BF16",
            )[0]
        )
        self.assertFalse(
            engine.compare_precision_results(
                baseline,
                outside,
                per_residue_tolerance=0.01,
                candidate_label="BF16",
            )[0]
        )

    @staticmethod
    def _alignment_validation_rows(count, changed_count=0):
        baseline = [
            (index, index + 1, 100.0, 100, 200.0, 100)
            for index in range(count)
        ]
        candidate = list(baseline)
        for index in range(changed_count):
            candidate[index] = (
                index,
                index + 1,
                104.0,
                104,
                192.0,
                96,
            )
        return baseline, candidate

    def test_bf16_clean_pass_keeps_per_residue_score_tolerance(self):
        baseline = [(0, 1, 100.0, 100, 200.0, 100)]
        at_limit = [(0, 1, 101.0, 100, 199.0, 100)]
        over_limit = [(0, 1, 101.001, 100, 200.0, 100)]

        accepted = engine.compare_bf16_precision_results(baseline, at_limit)
        rejected = engine.compare_bf16_precision_results(baseline, over_limit)

        self.assertTrue(accepted.accepted)
        self.assertFalse(accepted.warning)
        self.assertFalse(rejected.accepted)
        self.assertIn("global score tolerance", rejected.reason)

    def test_bf16_changed_pair_ceiling_is_strict(self):
        baseline, candidate = self._alignment_validation_rows(2048, 63)
        warning = engine.compare_bf16_precision_results(baseline, candidate)
        self.assertTrue(warning.accepted)
        self.assertTrue(warning.warning)
        self.assertEqual(warning.changed_count, 63)

        baseline, candidate = self._alignment_validation_rows(2048, 64)
        rejected = engine.compare_bf16_precision_results(baseline, candidate)
        self.assertFalse(rejected.accepted)
        self.assertIn("64/2048", rejected.reason)

    def test_bf16_changed_pair_limits_are_strict_and_check_both_modes(self):
        baseline = [(0, 1, 100.0, 100, 200.0, 100)] * 33
        baseline = [
            (index, index + 1, *row[2:]) for index, row in enumerate(baseline)
        ]
        below = list(baseline)
        below[0] = (0, 1, 104.999, 104, 190.002, 100)
        warning = engine.compare_bf16_precision_results(baseline, below)
        self.assertTrue(warning.accepted)
        self.assertTrue(warning.warning)

        length_at_limit = list(baseline)
        length_at_limit[0] = (0, 1, 100.0, 105, 200.0, 100)
        self.assertFalse(
            engine.compare_bf16_precision_results(
                baseline, length_at_limit
            ).accepted
        )

        local_score_at_limit = list(baseline)
        local_score_at_limit[0] = (0, 1, 100.0, 101, 190.0, 100)
        rejected = engine.compare_bf16_precision_results(
            baseline, local_score_at_limit
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("local score changed", rejected.reason)

    def test_bf16_zero_baselines_require_exact_zero(self):
        baseline = [(index, index + 1, 0.0, 0, 0.0, 100) for index in range(33)]
        unchanged_nonzero_score = list(baseline)
        unchanged_nonzero_score[0] = (0, 1, 0.001, 0, 0.0, 100)
        self.assertFalse(
            engine.compare_bf16_precision_results(
                baseline, unchanged_nonzero_score
            ).accepted
        )

        candidate = list(baseline)
        candidate[0] = (0, 1, 0.0, 0, 0.0, 101)
        accepted = engine.compare_bf16_precision_results(baseline, candidate)
        self.assertTrue(accepted.accepted)
        self.assertTrue(accepted.warning)

        nonzero_score = list(candidate)
        nonzero_score[0] = (0, 1, 0.001, 0, 0.0, 101)
        self.assertFalse(
            engine.compare_bf16_precision_results(
                baseline, nonzero_score
            ).accepted
        )

        changed_zero_length = list(baseline)
        changed_zero_length[0] = (0, 1, 0.0, 1, 0.0, 100)
        self.assertFalse(
            engine.compare_bf16_precision_results(
                baseline, changed_zero_length
            ).accepted
        )

    def test_bf16_warning_reports_each_extreme_metric(self):
        baseline, candidate = self._alignment_validation_rows(33, 1)
        result = engine.compare_bf16_precision_results(baseline, candidate)
        warning = engine.format_bf16_validation_warning(
            result,
            context="BF16 tiled on Test GPU",
            identity_label="pair",
        )

        self.assertIn("1/33 pairs", warning)
        self.assertIn("failure begins at 2/33", warning)
        for metric in (
            "global length",
            "local length",
            "global score",
            "local score",
        ):
            self.assertIn(f"worst {metric}", warning)
        self.assertEqual(warning.count("pair (0, 1)"), 4)

    def test_bf16_validation_rejects_invalid_result_sets(self):
        baseline = [(0, 1, 100.0, 100, 200.0, 100)]
        missing = []
        nonfinite = [(0, 1, np.nan, 100, 200.0, 100)]
        duplicate = baseline + baseline

        self.assertFalse(
            engine.compare_bf16_precision_results(baseline, missing).accepted
        )
        self.assertFalse(
            engine.compare_bf16_precision_results(baseline, nonfinite).accepted
        )
        self.assertFalse(
            engine.compare_bf16_precision_results(duplicate, duplicate).accepted
        )

    def test_align_bf16_warning_identifies_device_and_variant(self):
        device = align.Hardware_Utils.DeviceCandidate(
            "cuda:0", "Test GPU", torch.device("cuda:0"), "cuda"
        )
        baseline, candidate = self._alignment_validation_rows(33, 1)
        output = io.StringIO()
        with mock.patch.object(
            align.Hardware_Utils,
            "get_available_devices",
            return_value=[device],
        ), mock.patch.object(
            align.Hardware_Utils,
            "resolve_device_selection",
            return_value=device,
        ), mock.patch.object(
            align, "bf16_accelerator_support", return_value=(True, "supported")
        ), mock.patch.object(
            align, "_execution_variants", return_value=["scalar"]
        ), mock.patch.object(
            align,
            "_run_accelerated_pipeline",
            side_effect=[baseline, candidate],
        ), mock.patch.object(
            align, "_release_alignment_device_cache"
        ), redirect_stdout(output):
            precision = align._resolve_active_matmul_precision(
                "bf16",
                None,
                [(index, index + 1, "a", "b") for index in range(33)],
                1,
                mock.Mock(path="unused.h5"),
                [100] * 34,
            )

        self.assertEqual(precision, "bf16")
        self.assertIn("WARNING: BF16 scalar on Test GPU", output.getvalue())
        self.assertIn("worst global length", output.getvalue())

    def test_align_manual_bf16_fails_at_proportional_ceiling(self):
        device = align.Hardware_Utils.DeviceCandidate(
            "cuda:0", "Test GPU", torch.device("cuda:0"), "cuda"
        )
        baseline, candidate = self._alignment_validation_rows(32, 1)
        with mock.patch.object(
            align.Hardware_Utils,
            "get_available_devices",
            return_value=[device],
        ), mock.patch.object(
            align.Hardware_Utils,
            "resolve_device_selection",
            return_value=device,
        ), mock.patch.object(
            align, "bf16_accelerator_support", return_value=(True, "supported")
        ), mock.patch.object(
            align, "_execution_variants", return_value=["scalar"]
        ), mock.patch.object(
            align,
            "_run_accelerated_pipeline",
            side_effect=[baseline, candidate],
        ), mock.patch.object(
            align, "_release_alignment_device_cache"
        ), redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            ValueError, "1/32"
        ):
            align._resolve_active_matmul_precision(
                "bf16",
                None,
                [(index, index + 1, "a", "b") for index in range(32)],
                1,
                mock.Mock(path="unused.h5"),
                [100] * 33,
            )

    def test_network_injection_bf16_warning_keeps_candidate_eligible(self):
        device = injection.Hardware_Utils.DeviceCandidate(
            "cuda:0", "Test GPU", torch.device("cuda:0"), "cuda"
        )
        baseline, candidate = self._alignment_validation_rows(33, 1)
        memory = mock.Mock(free_bytes=8 << 30, total_bytes=10 << 30)
        estimate = mock.Mock(
            feasible=True,
            projected_peak_bytes=4 << 30,
            safe_peak_bytes=6 << 30,
            reason="safe",
        )

        def execute(*_args, **kwargs):
            timer = kwargs.get("benchmark_timer")
            if timer is not None:
                timer.start()
                timer.stop()
                return []
            return candidate

        output = io.StringIO()
        with mock.patch.object(
            injection.Hardware_Utils,
            "get_available_devices",
            return_value=[device],
        ), mock.patch.object(
            injection.Hardware_Utils,
            "resolve_device_selection",
            return_value=device,
        ), mock.patch.object(
            injection.Hardware_Utils, "release_device_cache"
        ), mock.patch.object(
            injection, "bf16_accelerator_support", return_value=(True, "supported")
        ), mock.patch.object(
            injection, "_execution_variants", return_value=["scalar"]
        ), mock.patch.object(
            injection, "_lane_candidates", return_value=[1]
        ), mock.patch.object(
            injection, "_benchmark_half_sizes", return_value=(4096, 256)
        ), mock.patch.object(
            injection, "cuda_memory_plan", return_value=memory
        ), mock.patch.object(
            injection, "estimate_cuda_working_set", return_value=estimate
        ), mock.patch.object(
            injection, "_run_accelerated_pipeline", return_value=baseline
        ), mock.patch.object(
            injection, "_execute_injection_plan", side_effect=execute
        ), redirect_stdout(output):
            plans = injection._benchmark_injection_plans(
                [(index, index + 1, "a", "b") for index in range(33)],
                workers=1,
                input_h5="unused.h5",
                store=mock.Mock(),
                lengths=[100] * 34,
                matmul_precision="bf16",
            )

        self.assertTrue(plans)
        self.assertIn("WARNING: BF16 scalar on Test GPU", output.getvalue())

    def test_hdf5_precision_normalization_recognizes_bf16(self):
        self.assertEqual(_normalized_precision("bf16"), "bf16")
        self.assertEqual(_normalized_precision("bfloat16"), "bfloat16")

    def test_network_injection_forwards_inherited_bf16(self):
        matrix = np.zeros((2, 3), dtype=np.float32)
        with mock.patch.object(
            injection, "_shared_score_matrix", return_value=matrix
        ) as score:
            result = injection._compute_accelerated_matrix(
                (
                    1,
                    2,
                    np.ones((2, 4), np.float32),
                    np.ones((3, 4), np.float32),
                    torch.device("cpu"),
                    "bf16",
                )
            )
        self.assertEqual(result[:2], (1, 2))
        self.assertEqual(score.call_args.kwargs["precision"], "bf16")

    def test_ssearch_scalar_worker_forwards_explicit_bf16(self):
        matrix = np.zeros((2, 3), dtype=np.float32)
        with mock.patch.object(
            ssearch, "_shared_score_matrix", return_value=matrix
        ) as score:
            result = ssearch._compute_accelerated_search(
                (
                    1,
                    "target",
                    np.ones((2, 4), np.float32),
                    np.ones((3, 4), np.float32),
                    "local",
                    -2.0,
                    "alignment_length",
                    torch.device("cpu"),
                    "bf16",
                )
            )
        self.assertEqual(result[:2], (1, "target"))
        self.assertEqual(score.call_args.kwargs["precision"], "bf16")

    def test_align_partial_writer_records_canonical_bf16(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "batch_00000.h5")
            writer = align._PartialBatchWriter(
                path,
                "checksum",
                "model",
                [-2.0, 0.0],
                "bf16",
            )
            writer([(0, 1, 1.0, 2, 2.0, 2)])
            writer.publish()
            with h5py.File(path, "r") as hf:
                self.assertEqual(hf.attrs["matmul_precision"], "bf16")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_real_cuda_bf16_probe_and_score_smoke(self):
        device = torch.device("cuda:0")
        supported, reason = engine.bf16_accelerator_support(
            device, refresh=True
        )
        if not supported:
            self.skipTest(reason)
        rng = np.random.default_rng(20260905)
        left = rng.normal(size=(7, 32)).astype(np.float32)
        right = rng.normal(size=(9, 32)).astype(np.float32)
        score = engine.compute_score_matrix_torch(
            left, right, device, precision="bf16"
        )
        self.assertEqual(score.dtype, np.float32)
        self.assertTrue(np.isfinite(score).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_real_cuda_bf16_tiled_smoke(self):
        device = torch.device("cuda:0")
        supported, reason = engine.bf16_accelerator_support(device)
        if not supported:
            self.skipTest(reason)
        rng = np.random.default_rng(42)
        headers = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            arrays = [
                rng.normal(size=(rows, 32)).astype(np.float32)
                for rows in (5, 7, 9)
            ]
            with h5py.File(path, "w") as hf:
                group = hf.create_group("embeddings")
                for header, array in zip(headers, arrays):
                    group.create_dataset(header, data=array)
            store = engine.EmbeddingTileStore(path, headers, 0)
            results = engine.run_tiled_accelerator_pipeline(
                [(0, 1, "a", "b"), (0, 2, "a", "c")],
                store=store,
                lengths=[5, 7, 9],
                device=device,
                workers=2,
                lanes=1,
                alignment_callback=lambda args: (
                    args[0],
                    args[1],
                    float(np.sum(args[2])),
                    int(args[2].shape[0]),
                    float(np.sum(args[2])),
                    int(args[2].shape[1]),
                ),
                precision="bf16",
            )
        self.assertEqual({result[:2] for result in results}, {(0, 1), (0, 2)})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_align_real_cuda_bf16_blocking_validation(self):
        device = torch.device("cuda:0")
        supported, reason = engine.bf16_accelerator_support(device)
        if not supported:
            self.skipTest(reason)
        rng = np.random.default_rng(7)
        headers = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "embeddings.h5")
            with h5py.File(path, "w") as hf:
                group = hf.create_group("embeddings")
                for header, rows in zip(headers, (6, 7, 8)):
                    group.create_dataset(
                        header,
                        data=rng.normal(size=(rows, 32)).astype(np.float32),
                    )
            store = engine.EmbeddingTileStore(path, headers, 0)
            tasks = [(0, 1, "a", "b"), (0, 2, "a", "c")]
            with mock.patch.object(align, "DEVICE_SELECTION", "cuda:0"), \
                    mock.patch.object(align, "EXECUTION_MODE", "scalar"), \
                    mock.patch.object(align, "LOCAL_GAP_P", -2.0), \
                    mock.patch.object(align, "GLOBAL_GAP_P", 0.0):
                precision = align._resolve_active_matmul_precision(
                    "bf16", None, tasks, 2, store, [6, 7, 8]
                )
        self.assertEqual(precision, "bf16")


if __name__ == "__main__":
    unittest.main()
