# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for runtime-compiled fused GPU alignment and SSEARCH kernels."""

import os
import re
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import h5py
import numpy as np
import torch

import Embedding_Alignment_Engine as engine
from tools import Align_Similarity_Matrix as align_tool
from tools import Embedding_SSEARCH as ssearch_tool
from utilities import Align_GPU_Kernels
from utilities.Network_Kernels import PACKED_MAX_PAIRS, PACKED_MAX_PATH, global_local_scores
from utilities.Similarity_Matrices import similar_pair_table


class AlignGPUKernelsCompileOptionsTests(unittest.TestCase):
    def test_hip_compile_options_define_amd_platform_macro_and_war_barrier(self):
        hip_opts = Align_GPU_Kernels._compile_options(True, "gfx1100")
        self.assertIn("--offload-arch=gfx1100", hip_opts)
        self.assertIn("-ffp-contract=off", hip_opts)
        self.assertIn("-D__HIP_PLATFORM_AMD__=1", hip_opts)

        cuda_opts = Align_GPU_Kernels._compile_options(False, "sm_89")
        self.assertEqual(cuda_opts, ["--gpu-architecture=sm_89", "--fmad=false"])

        # Kernel source must route all ROCm/hiprtc builds to the portable shared-memory path
        # and synchronize after reading smem_warp[0] before returning from warp_sum_32.
        hip_guard = re.search(
            r"#if\s+([^\n]+)\s*\n\s*#define\s+SSN_ALIGN_HIP_PATH\b",
            Align_GPU_Kernels.KERNEL_SOURCE,
        )
        self.assertIsNotNone(hip_guard, "Missing SSN_ALIGN_HIP_PATH preprocessor guard")
        for macro in ("__HIP_PLATFORM_AMD__", "__HIPCC_RTC__", "__HIP__"):
            self.assertRegex(hip_guard.group(1), rf"defined\(\s*{macro}\s*\)")
        self.assertNotRegex(Align_GPU_Kernels.KERNEL_SOURCE, r"#ifdef\s+__HIP_PLATFORM_AMD__\b")
        warp_fn = re.search(
            r"warp_sum_32\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            Align_GPU_Kernels.KERNEL_SOURCE,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(warp_fn, "Missing warp_sum_32 definition")
        self.assertRegex(
            warp_fn.group("body"),
            r"float\s+(\w+)\s*=\s*smem_warp\[\s*0\s*\]\s*;\s*__syncthreads\(\s*\)\s*;\s*return\s+\1\s*;",
        )


class AlignGPUKernelsCPUHelperTests(unittest.TestCase):
    """CPU-always unit tests for Align_GPU_Kernels policy, sizing, and fallback helpers."""

    def test_kernels_disabled_and_align_kernels_short_circuit(self):
        var = Align_GPU_Kernels.DISABLE_ENVIRONMENT_VARIABLE
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(var, None)
            self.assertFalse(Align_GPU_Kernels.kernels_disabled())
        for enabled_val in ("", "1", "true", "yes", "on"):
            with self.subTest(value=enabled_val), mock.patch.dict(os.environ, {var: enabled_val}):
                self.assertFalse(Align_GPU_Kernels.kernels_disabled())
        for disabled_val in ("0", "off", "false", "no", " OFF ", "False"):
            with self.subTest(value=disabled_val), mock.patch.dict(os.environ, {var: disabled_val}):
                self.assertTrue(Align_GPU_Kernels.kernels_disabled())
        with mock.patch.dict(os.environ, {var: "0"}), \
                mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertIsNone(Align_GPU_Kernels.align_kernels(torch.device("cuda:0")))

    def test_fallback_reason_reports_device_disable_and_recorded_failure(self):
        self.assertIn("cpu", Align_GPU_Kernels.fallback_reason(torch.device("cpu")))
        self.assertIn("mps", Align_GPU_Kernels.fallback_reason("mps"))
        var = Align_GPU_Kernels.DISABLE_ENVIRONMENT_VARIABLE
        with mock.patch.dict(os.environ, {var: "0"}):
            reason = Align_GPU_Kernels.fallback_reason(torch.device("cuda:0"))
            self.assertIn(var, reason)
        with mock.patch.dict(os.environ, {var: "1"}), \
                mock.patch.dict(Align_GPU_Kernels._REASONS, {0: "RuntimeError: mock compile failed"}, clear=True):
            self.assertEqual(
                Align_GPU_Kernels.fallback_reason(torch.device("cuda:0")),
                "RuntimeError: mock compile failed",
            )
            self.assertIsNone(Align_GPU_Kernels.fallback_reason(torch.device("cuda:1")))

    def test_can_use_fused_search_validates_gaps_and_packed_length_bounds(self):
        tasks = [(0, "t0"), (1, "t1")]
        lengths = [64, 128]
        task_index = lambda t: t[0]

        self.assertFalse(Align_GPU_Kernels.can_use_fused_search([], 100, lengths, {"mode": "local", "gap": -2.0}, task_index))
        self.assertFalse(Align_GPU_Kernels.can_use_fused_search(tasks, 100, lengths, None, task_index))
        self.assertTrue(Align_GPU_Kernels.can_use_fused_search(tasks, 100, lengths, {"mode": "local", "gap": -2.0}, task_index))
        self.assertTrue(Align_GPU_Kernels.can_use_fused_search(tasks, 100, lengths, {"mode": "global", "gap": 0.0}, task_index))
        self.assertTrue(Align_GPU_Kernels.can_use_fused_search(tasks, 100, lengths, {"mode": "global", "gap": -0.5}, task_index))

        # Non-exact float32 gaps are rejected cleanly.
        self.assertFalse(Align_GPU_Kernels.can_use_fused_search(tasks, 100, lengths, {"mode": "local", "gap": -0.7}, task_index))
        self.assertFalse(Align_GPU_Kernels.can_use_fused_search(tasks, 100, lengths, {"mode": "global", "gap": -0.1}, task_index))

        # Invalid or overflowed sequence / path lengths are rejected cleanly.
        self.assertFalse(Align_GPU_Kernels.can_use_fused_search(tasks, 0, lengths, {"mode": "local", "gap": -2.0}, task_index))
        self.assertFalse(Align_GPU_Kernels.can_use_fused_search(tasks, PACKED_MAX_PAIRS + 1, lengths, {"mode": "local", "gap": -2.0}, task_index))
        self.assertFalse(Align_GPU_Kernels.can_use_fused_search(tasks, 100, [0, 0], {"mode": "local", "gap": -2.0}, task_index))
        self.assertFalse(
            Align_GPU_Kernels.can_use_fused_search(
                tasks, PACKED_MAX_PAIRS, [PACKED_MAX_PATH - PACKED_MAX_PAIRS + 1, 10],
                {"mode": "global", "gap": 0.0}, task_index,
            )
        )

    def test_choose_block_residues_respects_vram_streams_override_and_max_seq(self):
        # Huge VRAM clamps at DEFAULT_BLOCK_RESIDUES.
        large = Align_GPU_Kernels._choose_block_residues(32 * (1 << 30), 2, 200)
        self.assertEqual(large, Align_GPU_Kernels.DEFAULT_BLOCK_RESIDUES)

        # Tiny VRAM without override hits the 16 MiB per-stream floor -> side 2048.
        tiny = Align_GPU_Kernels._choose_block_residues(16 * (1 << 20), 2, 200)
        self.assertEqual(tiny, 2048)

        # Tiny matrix_budget_override hits the 4 MiB floor -> MIN_BLOCK_RESIDUES (1024).
        tiny_override = Align_GPU_Kernels._choose_block_residues(
            16 * (1 << 20), 2, 200, matrix_budget_override=1
        )
        self.assertEqual(tiny_override, Align_GPU_Kernels.MIN_BLOCK_RESIDUES)

        # Multiple streams divide the per-stream matrix budget.
        one_stream = Align_GPU_Kernels._choose_block_residues(512 * (1 << 20), 1, 100)
        two_streams = Align_GPU_Kernels._choose_block_residues(512 * (1 << 20), 2, 100)
        self.assertEqual(one_stream % 512, 0)
        self.assertEqual(two_streams % 512, 0)
        self.assertGreater(one_stream, two_streams)

        # matrix_budget_override caps per-stream budget (4 MiB -> side 1024).
        overridden = Align_GPU_Kernels._choose_block_residues(
            32 * (1 << 30), 1, 100, matrix_budget_override=4 * 1024 * 1024
        )
        self.assertEqual(overridden, 1024)

        # Never returns smaller than max_seq_len.
        self.assertEqual(
            Align_GPU_Kernels._choose_block_residues(16 * (1 << 20), 2, 12345),
            12345,
        )

    def test_active_task_block_ids_packs_hubs_first_and_respects_block_budget(self):
        # Empty tasks -> all-zero int32 array.
        empty = Align_GPU_Kernels._active_task_block_ids([], 5, [10] * 5, 16, 4, 1024)
        self.assertEqual(empty.dtype, np.int32)
        np.testing.assert_array_equal(empty, np.zeros(5, dtype=np.int32))

        # Sequences 8 and 9 are high-degree hubs connected to 0..5; sequences 6, 7 are inactive.
        # Each seq is 10 rows * 4 dim * 4 B = 160 B; per_block = 320 B fits exactly 2 seqs per block.
        n_seqs = 10
        lengths = [10] * n_seqs
        tasks = [(i, 8, f"s{i}", "s8") for i in range(6)] + [(i, 9, f"s{i}", "s9") for i in range(6)]
        block_ids = Align_GPU_Kernels._active_task_block_ids(tasks, n_seqs, lengths, 4, 4, 320)

        # Hubs (8, 9) are packed together into block 0 in ascending order.
        self.assertEqual((int(block_ids[8]), int(block_ids[9])), (0, 0))
        # Remaining active sequences (0..5) fill blocks 1, 2, 3 in ascending pairs.
        np.testing.assert_array_equal(block_ids[:6], [1, 1, 2, 2, 3, 3])
        # Inactive sequences remain at default 0 and never consume space in active blocks.
        np.testing.assert_array_equal(block_ids[6:8], [0, 0])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class AlignGPUKernelsTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            clear_cublas = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_cublas is not None:
                clear_cublas()
            torch.cuda.empty_cache()

    def _make_store(self, temp_dir, lengths, dim=64, seed=20261123, dtype=np.float32, host_cache=None):
        rng = np.random.default_rng(seed)
        headers = [f"seq_{i}" for i in range(len(lengths))]
        path = os.path.join(temp_dir, "embs.h5")
        with h5py.File(path, "w") as hf:
            grp = hf.create_group("embeddings")
            for header, length in zip(headers, lengths):
                grp.create_dataset(
                    header,
                    data=rng.standard_normal((length, dim)).astype(dtype),
                )
        cache_arg = "auto" if host_cache is None else host_cache
        store = engine.EmbeddingTileStore(path, headers, host_cache_setting=cache_arg)
        return path, headers, store

    def test_kernel_compilation_and_self_check(self):
        kernels = Align_GPU_Kernels.align_kernels(torch.device("cuda:0"))
        self.assertIsNotNone(kernels)
        self.assertIsNone(Align_GPU_Kernels.fallback_reason(torch.device("cuda:0")))
        kernels.self_check()

    def test_fused_dense_and_sparse_match_reference_default_and_custom_gaps(self):
        lengths = [1, 15, 64, 128, 145, 270]
        with tempfile.TemporaryDirectory() as temp_dir:
            _path, headers, store = self._make_store(temp_dir, lengths, dim=64)
            all_tasks = [
                (i, j, headers[i], headers[j])
                for i in range(len(lengths))
                for j in range(i + 1, len(lengths))
            ]
            # Sparse subset (<20% cell density) exercises zero-padding row-packed path.
            sparse_tasks = [all_tasks[0], all_tasks[4], all_tasks[-1]]

            for tasks in (all_tasks, sparse_tasks):
                for global_gap, local_gap in ((0.0, -2.0), (-0.7, -1.3)):
                    with mock.patch.object(align_tool, "GLOBAL_GAP_P", global_gap), \
                            mock.patch.object(align_tool, "LOCAL_GAP_P", local_gap):
                        fused = engine.run_tiled_accelerator_pipeline(
                            tasks,
                            store=store,
                            lengths=lengths,
                            device=torch.device("cuda:0"),
                            workers=2,
                            lanes=2,
                            alignment_callback=align_tool.calculate_alignment_data,
                            batch_alignment_callback=align_tool.calculate_alignment_batch,
                            precision="float32",
                            gpu_dp_gaps=(global_gap, local_gap),
                        )
                        ref = engine.run_tiled_accelerator_pipeline(
                            tasks,
                            store=store,
                            lengths=lengths,
                            device=torch.device("cuda:0"),
                            workers=2,
                            lanes=1,
                            alignment_callback=align_tool.calculate_alignment_data,
                            batch_alignment_callback=align_tool.calculate_alignment_batch,
                            precision="float32",
                            gpu_dp_gaps=None,
                        )
                    ok, reason = engine.compare_precision_results(
                        ref, fused, per_residue_tolerance=1e-4, candidate_label="fused_gpu"
                    )
                    self.assertTrue(ok, f"Mismatch for gaps={(global_gap, local_gap)}: {reason}")

    def test_fused_detects_non_finite_embeddings(self):
        lengths = [10, 12]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "bad.h5")
            with h5py.File(path, "w") as hf:
                grp = hf.create_group("embeddings")
                arr0 = np.ones((10, 16), dtype=np.float32)
                arr0[3, 2] = np.nan
                grp.create_dataset("s0", data=arr0)
                grp.create_dataset("s1", data=np.ones((12, 16), dtype=np.float32))
            store = engine.EmbeddingTileStore(path, ["s0", "s1"], host_cache_setting=0)
            with self.assertRaises(FloatingPointError):
                engine.run_tiled_accelerator_pipeline(
                    [(0, 1, "s0", "s1")],
                    store=store,
                    lengths=lengths,
                    device=torch.device("cuda:0"),
                    workers=1,
                    lanes=1,
                    alignment_callback=align_tool.calculate_alignment_data,
                    batch_alignment_callback=align_tool.calculate_alignment_batch,
                    precision="float32",
                    gpu_dp_gaps=(0.0, -2.0),
                )

    def test_fused_fixed_query_search_matches_reference(self):
        rng = np.random.default_rng(999)
        alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        lengths = [12, 64, 129, 260, 40]
        q_len = 135
        q_seq = "".join(rng.choice(alphabet, size=q_len))
        t_seqs = ["".join(rng.choice(alphabet, size=L)) for L in lengths]
        q_emb = rng.standard_normal((q_len, 64)).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            _path, headers, store = self._make_store(temp_dir, lengths, dim=64, host_cache=0)
            for mode, gap, norm_mode in (
                ("local", -2.0, "shorter_sequence"),
                ("global", 0.0, "alignment_length"),
            ):
                tasks = [
                    (idx, headers[idx], headers[idx], q_emb, mode, gap, norm_mode, q_seq, t_seqs[idx], "BLOSUM62")
                    for idx in range(len(lengths))
                ]
                search_cfg = {
                    "mode": mode,
                    "gap": gap,
                    "norm_mode": norm_mode,
                    "query_codes": ssearch_tool.encode_identity_residues(q_seq, ssearch_tool.QUERY_UNMATCHED_CODE),
                    "similar_table": similar_pair_table("BLOSUM62"),
                    "encode_target_fn": lambda t, _L: ssearch_tool.encode_identity_residues(
                        t[8], ssearch_tool.TARGET_UNMATCHED_CODE
                    ),
                    "make_row_fn": ssearch_tool._search_row,
                    "mismatched_error_fn": ssearch_tool._mismatched_matrix_error,
                }
                fused = engine.run_fixed_query_cuda_pipeline(
                    tasks,
                    query_embedding=q_emb,
                    store=store,
                    lengths=lengths,
                    device=torch.device("cuda:0"),
                    workers=2,
                    lanes=2,
                    batch_alignment_callback=ssearch_tool.finish_search_batch,
                    precision="float32",
                    gpu_search_config=search_cfg,
                )
                ref = engine.run_fixed_query_cuda_pipeline(
                    tasks,
                    query_embedding=q_emb,
                    store=store,
                    lengths=lengths,
                    device=torch.device("cuda:0"),
                    workers=2,
                    lanes=1,
                    batch_alignment_callback=ssearch_tool.finish_search_batch,
                    precision="float32",
                    gpu_search_config=None,
                )
                fused_by_id = {row["index"]: row for row in fused}
                ref_by_id = {row["index"]: row for row in ref}
                self.assertEqual(fused_by_id.keys(), ref_by_id.keys())
                for idx in ref_by_id:
                    f_row = fused_by_id[idx]
                    r_row = ref_by_id[idx]
                    self.assertEqual(f_row["aln_len"], r_row["aln_len"])
                    self.assertAlmostEqual(f_row["identity"], r_row["identity"], places=6)
                    self.assertAlmostEqual(f_row["similarity"], r_row["similarity"], places=6)
                    self.assertAlmostEqual(f_row["gaps"], r_row["gaps"], places=6)
                    self.assertLess(
                        abs(f_row["raw_score"] - r_row["raw_score"]) / max(1, r_row["aln_len"]),
                        1e-4,
                    )

    def test_fused_multitile_fp16_releases_prior_tile_under_tight_cap(self):
        from dataclasses import replace

        lengths = [160] * 12
        dim = 256
        with tempfile.TemporaryDirectory() as temp_dir:
            _path, headers, store = self._make_store(
                temp_dir, lengths, dim=dim, dtype=np.float16, host_cache="auto"
            )
            tasks = [
                (i, j, headers[i], headers[j])
                for i in range(len(lengths))
                for j in range(i + 1, len(lengths))
            ]
            # Force 3 blocks (per_block = 4 seqs -> 6 tiles across (0,0)..(2,2)) so multiple
            # tile transitions execute in one phase under a tight allocator cap.
            seq_f32_bytes = lengths[0] * dim * 4
            tile_cache_bytes = seq_f32_bytes * 8
            dev = torch.device("cuda:0")
            base_plan = engine.accelerator_memory_plan(dev, lanes=2, compute_element_bytes=4)
            tight_plan = replace(
                base_plan,
                tile_cache_bytes=tile_cache_bytes,
                allocator_cap_bytes=tile_cache_bytes + 512 * 1024 * 1024,
            )
            fused = engine.run_tiled_accelerator_pipeline(
                tasks,
                store=store,
                lengths=lengths,
                device=dev,
                workers=2,
                lanes=2,
                alignment_callback=align_tool.calculate_alignment_data,
                batch_alignment_callback=align_tool.calculate_alignment_batch,
                precision="float32",
                memory_plan_override=tight_plan,
                gpu_dp_gaps=(0.0, -2.0),
            )
            ref = engine.run_tiled_accelerator_pipeline(
                tasks,
                store=store,
                lengths=lengths,
                device=dev,
                workers=2,
                lanes=1,
                alignment_callback=align_tool.calculate_alignment_data,
                batch_alignment_callback=align_tool.calculate_alignment_batch,
                precision="float32",
                gpu_dp_gaps=None,
            )
            ok, reason = engine.compare_precision_results(
                ref, fused, per_residue_tolerance=1e-4, candidate_label="fused_multitile_fp16"
            )
            self.assertTrue(ok, reason)

    def test_fused_sparse_multichunk_2lanes_matches_reference(self):
        # Regression test for cross-stream t_cat race: 1 row against 240 targets with D=1152
        # forces sparse density (<20%) and multiple <=64 MiB t_cat chunks across 2 async streams.
        rng = np.random.default_rng(20261123)
        lengths = [309] + rng.integers(289, 312, size=240).tolist()
        dim = 1152
        with tempfile.TemporaryDirectory() as temp_dir:
            _path, headers, store = self._make_store(
                temp_dir, lengths, dim=dim, dtype=np.float16, host_cache="auto"
            )
            tasks = [(0, j, headers[0], headers[j]) for j in range(1, len(lengths))]
            dev = torch.device("cuda:0")
            fused = engine.run_tiled_accelerator_pipeline(
                tasks,
                store=store,
                lengths=lengths,
                device=dev,
                workers=2,
                lanes=2,
                alignment_callback=align_tool.calculate_alignment_data,
                batch_alignment_callback=align_tool.calculate_alignment_batch,
                precision="float32",
                gpu_dp_gaps=(0.0, -2.0),
            )
            ref = engine.run_tiled_accelerator_pipeline(
                tasks,
                store=store,
                lengths=lengths,
                device=dev,
                workers=2,
                lanes=1,
                alignment_callback=align_tool.calculate_alignment_data,
                batch_alignment_callback=align_tool.calculate_alignment_batch,
                precision="float32",
                gpu_dp_gaps=None,
            )
            ok, reason = engine.compare_precision_results(
                ref, fused, per_residue_tolerance=1e-4, candidate_label="fused_sparse_2lanes"
            )
            self.assertTrue(ok, reason)

    def test_fused_sparse_multirow_slab_multitile_2lanes_matches_reference(self):
        # Exercises tile-wide 2D-shelf sparse slabs across 2 async lanes with mixed single-target
        # zero-copy rows and multi-target b_buf rows across irregular sequence lengths and 3 blocks.
        from dataclasses import replace

        rng = np.random.default_rng(20261124)
        n_seqs = 60
        lengths = rng.integers(80, 260, size=n_seqs).tolist()
        dim = 256
        with tempfile.TemporaryDirectory() as temp_dir:
            _path, headers, store = self._make_store(
                temp_dir, lengths, dim=dim, dtype=np.float16, host_cache="auto"
            )
            pair_set = set()
            for u in range(n_seqs):
                deg = 1 if u < 15 else int(rng.integers(2, 7))
                partners = rng.choice(n_seqs, size=deg, replace=False)
                for v in partners:
                    v = int(v)
                    if u != v:
                        a, b = (u, v) if u < v else (v, u)
                        pair_set.add((a, b))
            tasks = [(u, v, headers[u], headers[v]) for u, v in sorted(pair_set)]
            dev = torch.device("cuda:0")
            base_plan = engine.accelerator_memory_plan(dev, lanes=2, compute_element_bytes=4)
            # Force ~3 blocks so sparse slabs run across both diagonal and off-diagonal tiles.
            per_block_bytes = int(sum(lengths) * dim * 4 // 3)
            tight_plan = replace(base_plan, tile_cache_bytes=max(1, per_block_bytes * 2))
            fused = engine.run_tiled_accelerator_pipeline(
                tasks,
                store=store,
                lengths=lengths,
                device=dev,
                workers=2,
                lanes=2,
                alignment_callback=align_tool.calculate_alignment_data,
                batch_alignment_callback=align_tool.calculate_alignment_batch,
                precision="float32",
                memory_plan_override=tight_plan,
                gpu_dp_gaps=(0.0, -2.0),
            )
            ref = engine.run_tiled_accelerator_pipeline(
                tasks,
                store=store,
                lengths=lengths,
                device=dev,
                workers=2,
                lanes=1,
                alignment_callback=align_tool.calculate_alignment_data,
                batch_alignment_callback=align_tool.calculate_alignment_batch,
                precision="float32",
                gpu_dp_gaps=None,
            )
            self.assertEqual(len(fused), len(tasks))
            ok, reason = engine.compare_precision_results(
                ref, fused, per_residue_tolerance=1e-4, candidate_label="fused_sparse_slab"
            )
            self.assertTrue(ok, reason)

    def test_hip_shared_memory_barrier_path_passes_self_check(self):
        # Compile the exact HIP shared-memory lockstep path and execute self_check()
        # (mixed lengths 1..260, partial blocks, local + global + f64-gap + search)
        # on GPU to prove zero __syncthreads() divergence deadlocks, WAR-safe warp_sum_32,
        # and exact numerical parity.
        dev = torch.device("cuda:0")
        hip_source = "#define __HIP_PLATFORM_AMD__ 1\n" + Align_GPU_Kernels.KERNEL_SOURCE
        with mock.patch.object(Align_GPU_Kernels, "KERNEL_SOURCE", hip_source):
            hip_kernels = Align_GPU_Kernels.AlignKernels(dev, read_cache=False)
            hip_kernels.self_check()

    def test_oom_during_fused_pipeline_synchronizes_cuda_before_cleanup(self):
        # BLK-1: An exception/OOM while prior stream work is still inflight must synchronize
        # CUDA before dropping packed tile views, clearing workspaces, or emptying cache.
        lengths = [100] * 16
        dim = 64
        with tempfile.TemporaryDirectory() as temp_dir:
            _path, headers, store = self._make_store(temp_dir, lengths, dim=dim, host_cache="auto")
            tasks = [
                (i, j, headers[i], headers[j])
                for i in range(len(lengths))
                for j in range(i + 1, len(lengths))
            ]
            dev = torch.device("cuda:0")
            kernels = Align_GPU_Kernels.align_kernels(dev)
            self.assertIsNotNone(kernels)

            real_launch = kernels.launch
            real_sync = torch.cuda.synchronize
            real_empty_cache = torch.cuda.empty_cache
            launch_calls = 0
            events = []

            def failing_launch(name, grid, block, args, stream=None):
                nonlocal launch_calls
                launch_calls += 1
                # Job 0 launches align_exp_stats (1) and DP (2) and stays inflight on stream 0;
                # fail on Job 1's first kernel launch (3) on stream 1.
                if launch_calls == 3:
                    events.append("oom")
                    raise torch.cuda.OutOfMemoryError("simulated OOM during stream 1 launch")
                return real_launch(name, grid, block, args, stream=stream)

            def tracked_sync(device_arg=None):
                events.append("sync")
                return real_sync(device_arg)

            def tracked_empty_cache():
                events.append("empty_cache")
                return real_empty_cache()

            with mock.patch.object(kernels, "launch", side_effect=failing_launch), \
                    mock.patch.object(torch.cuda, "synchronize", side_effect=tracked_sync), \
                    mock.patch.object(torch.cuda, "empty_cache", side_effect=tracked_empty_cache):
                with self.assertRaises(torch.cuda.OutOfMemoryError):
                    Align_GPU_Kernels.run_fused_gpu_alignment_pipeline(
                        tasks,
                        store=store,
                        lengths=lengths,
                        device=dev,
                        lanes=2,
                        global_gap=0.0,
                        local_gap=-2.0,
                        precision="float32",
                        matrix_budget_override=4 * 1024 * 1024,
                        cuda_matmul_precision_ctx=engine.cuda_matmul_precision,
                        precision_compute_dtype_fn=engine.precision_compute_dtype,
                        precision_element_bytes_fn=engine.precision_element_bytes,
                        accelerator_memory_plan_fn=engine.accelerator_memory_plan,
                        partition_tiles_fn=engine._partition_tiles,
                    )

            self.assertIn("oom", events)
            oom_idx = events.index("oom")
            post_oom = events[oom_idx + 1:]
            # Both per-tile finally (for inflight) and outer finally (before empty_cache) must sync.
            self.assertGreaterEqual(post_oom.count("sync"), 2, f"Expected syncs after OOM: {events}")
            self.assertIn("empty_cache", post_oom)
            self.assertLess(post_oom.index("sync"), post_oom.index("empty_cache"))

    def test_duplicate_and_symmetric_pairs_match_cpu_reference(self):
        # HI-2: Duplicate and symmetric (si, sj)/(sj, si) pairs in dense blocks must not
        # double-transform the same mat submatrix in-place inside align_exp_stats.
        lengths = [120] * 8 + [160] * 6
        dim = 64
        with tempfile.TemporaryDirectory() as temp_dir:
            _path, headers, store = self._make_store(temp_dir, lengths, dim=dim, host_cache="auto")
            base_pairs = [
                (i, j)
                for i in range(len(lengths))
                for j in range(i + 1, len(lengths))
            ]
            # Add diagonal duplicates, cross-block duplicates, symmetric cross-block pairs,
            # symmetric diagonal pairs, and duplicate self-pairs.
            extra_pairs = [
                (0, 1), (0, 1), (1, 0),       # diagonal block dup + symmetric
                (0, 10), (0, 10), (10, 0),    # cross-block dup + symmetric (colliding canonical cell)
                (2, 2), (2, 2),               # duplicate self-pair
                (9, 12), (12, 9),             # second diagonal block symmetric
            ]
            all_pairs = base_pairs + extra_pairs
            tasks = [(i, j, headers[i], headers[j]) for i, j in all_pairs]

            dev = torch.device("cuda:0")
            fused = Align_GPU_Kernels.run_fused_gpu_alignment_pipeline(
                tasks,
                store=store,
                lengths=lengths,
                device=dev,
                lanes=2,
                global_gap=0.0,
                local_gap=-2.0,
                precision="float32",
                matrix_budget_override=4 * 1024 * 1024,  # block_res=1024 -> 2 dense blocks
                cuda_matmul_precision_ctx=engine.cuda_matmul_precision,
                precision_compute_dtype_fn=engine.precision_compute_dtype,
                precision_element_bytes_fn=engine.precision_element_bytes,
                accelerator_memory_plan_fn=engine.accelerator_memory_plan,
                partition_tiles_fn=engine._partition_tiles,
            )

            self.assertEqual(len(fused), len(tasks))
            # Build exact CPU reference scores for every unique ordered pair (si, sj).
            ref_by_pair = {}
            for si, sj in set(all_pairs):
                ti = torch.nn.functional.normalize(
                    torch.from_numpy(store.get(si)).double(), p=2, dim=-1
                )
                tj = torch.nn.functional.normalize(
                    torch.from_numpy(store.get(sj)).double(), p=2, dim=-1
                )
                cos = torch.mm(ti, tj.T).clamp(-1.0, 1.0)
                sim = torch.exp(-(1.0 - cos))
                rm = sim.mean(dim=1, keepdim=True)
                rs = sim.std(dim=1, keepdim=True, correction=0)
                cm = sim.mean(dim=0, keepdim=True)
                cs = sim.std(dim=0, keepdim=True, correction=0)
                sc = ((((sim - rm) / (rs + 1e-8)) + ((sim - cm) / (cs + 1e-8))) / 2.0).float().numpy()
                rgs, rgl, rls, rll = global_local_scores(sc, 0.0, -2.0)
                ref_by_pair[(si, sj)] = (float(rls), int(rll), float(rgs), int(rgl))

            from collections import Counter
            self.assertEqual(
                Counter((r[0], r[1]) for r in fused),
                Counter(all_pairs),
            )
            for si, sj, l_s, l_l, g_s, g_l in fused:
                rls, rll, rgs, rgl = ref_by_pair[(si, sj)]
                self.assertEqual(int(l_l), rll, f"local length mismatch on ({si}, {sj})")
                self.assertEqual(int(g_l), rgl, f"global length mismatch on ({si}, {sj})")
                self.assertLess(
                    abs(float(l_s) - rls) / max(1, rll),
                    1e-4,
                    f"local score mismatch on ({si}, {sj}): {l_s} vs {rls}",
                )
                self.assertLess(
                    abs(float(g_s) - rgs) / max(1, rgl),
                    1e-4,
                    f"global score mismatch on ({si}, {sj}): {g_s} vs {rgs}",
                )

    def test_mixed_short_long_block_uses_ragged_dp_scratch_and_matches_reference(self):
        # HI-8: Mixed short + long sequence block (max_c > 128) must allocate ragged tot_r
        # DP scratch (far smaller than num_p * max_r), while max_c <= 128 allocates size 1.
        lengths = [6] * 80 + [500, 500]
        dim = 32
        with tempfile.TemporaryDirectory() as temp_dir:
            _path, headers, store = self._make_store(temp_dir, lengths, dim=dim, host_cache="auto")
            tasks = [
                (i, j, headers[i], headers[j])
                for i in range(len(lengths))
                for j in range(i + 1, len(lengths))
            ]
            dev = torch.device("cuda:0")
            kernels = Align_GPU_Kernels.align_kernels(dev)
            self.assertIsNotNone(kernels)

            num_p = len(tasks)
            max_r = max(lengths[i] for i, _, _, _ in tasks)
            tot_r = sum(lengths[i] for i, _, _, _ in tasks)
            self.assertGreater(num_p * max_r, tot_r * 10)

            for global_gap, local_gap, expected_kernel in (
                (0.0, -2.0, "align_dp_f32_k4"),
                (-0.7, -1.3, "align_dp_f64gap_k4"),
            ):
                seen_scratch_numels = []
                real_launch = kernels.launch

                def spy_launch(name, grid, block, args, stream=None):
                    if name == expected_kernel:
                        # args[13..16] are sc_g, sc_gl, sc_l, sc_ll
                        seen_scratch_numels.append(int(args[13].numel()))
                    return real_launch(name, grid, block, args, stream=stream)

                with mock.patch.object(kernels, "launch", side_effect=spy_launch), \
                        mock.patch.object(align_tool, "GLOBAL_GAP_P", global_gap), \
                        mock.patch.object(align_tool, "LOCAL_GAP_P", local_gap):
                    fused = engine.run_tiled_accelerator_pipeline(
                        tasks,
                        store=store,
                        lengths=lengths,
                        device=dev,
                        workers=2,
                        lanes=1,
                        alignment_callback=align_tool.calculate_alignment_data,
                        batch_alignment_callback=align_tool.calculate_alignment_batch,
                        precision="float32",
                        gpu_dp_gaps=(global_gap, local_gap),
                    )
                    ref = engine.run_tiled_accelerator_pipeline(
                        tasks,
                        store=store,
                        lengths=lengths,
                        device=dev,
                        workers=2,
                        lanes=1,
                        alignment_callback=align_tool.calculate_alignment_data,
                        batch_alignment_callback=align_tool.calculate_alignment_batch,
                        precision="float32",
                        gpu_dp_gaps=None,
                    )

                self.assertTrue(seen_scratch_numels)
                for sc_numel in seen_scratch_numels:
                    self.assertGreaterEqual(sc_numel, tot_r)
                    self.assertLess(sc_numel, num_p * max_r // 10)

                ok, reason = engine.compare_precision_results(
                    ref, fused, per_residue_tolerance=1e-4, candidate_label=expected_kernel
                )
                self.assertTrue(ok, f"Mismatch for {expected_kernel}: {reason}")

            # Short-only block (max_c <= 128) should only allocate 1-element scratch.
            short_tasks = [
                (i, j, headers[i], headers[j])
                for i in range(10)
                for j in range(i + 1, 10)
            ]
            short_scratch_numels = []
            real_launch = kernels.launch

            def spy_short(name, grid, block, args, stream=None):
                if name == "align_dp_f32_k4":
                    short_scratch_numels.append(int(args[13].numel()))
                return real_launch(name, grid, block, args, stream=stream)

            with mock.patch.object(kernels, "launch", side_effect=spy_short):
                Align_GPU_Kernels.run_fused_gpu_alignment_pipeline(
                    short_tasks,
                    store=store,
                    lengths=lengths,
                    device=dev,
                    lanes=1,
                    global_gap=0.0,
                    local_gap=-2.0,
                    precision="float32",
                    cuda_matmul_precision_ctx=engine.cuda_matmul_precision,
                    precision_compute_dtype_fn=engine.precision_compute_dtype,
                    precision_element_bytes_fn=engine.precision_element_bytes,
                    accelerator_memory_plan_fn=engine.accelerator_memory_plan,
                    partition_tiles_fn=engine._partition_tiles,
                )
            self.assertEqual(short_scratch_numels, [1])

    def test_upload_uses_device_context_explicit_stream_and_syncs_events_on_error(self):
        # HI-5 & M7: _upload_and_normalize_indices must enter torch.cuda.device(device),
        # pass explicit current_stream(device) to Event.record(), and synchronize pending
        # pinned-buffer events in finally when an exception interrupts the upload loop.
        lengths = [64, 64, 64]
        dim = 64
        with tempfile.TemporaryDirectory() as temp_dir:
            _path, _headers, store = self._make_store(temp_dir, lengths, dim=dim, host_cache="auto")
            dev = torch.device("cuda:0")
            real_get = store.get
            real_event_cls = torch.cuda.Event
            real_device_ctx = torch.cuda.device
            entered_devices = []
            recorded_streams = []
            synced_events = []

            class SpyEvent:
                def __init__(self):
                    self._ev = real_event_cls()
                    self.synced = False

                def record(self, stream=None):
                    recorded_streams.append(stream)
                    return self._ev.record(stream)

                def synchronize(self):
                    self.synced = True
                    synced_events.append(self)
                    return self._ev.synchronize()

            class SpyDevice(real_device_ctx):
                def __init__(self, d):
                    entered_devices.append(d)
                    super().__init__(d)

            def failing_get(idx, group=None):
                if idx == 2:
                    raise RuntimeError("simulated HDF5 read failure")
                return real_get(idx, group)

            # Force chunk_rows = 64 so seq 0 flushes into buf_events[0] before seq 2 raises.
            small_chunk_bytes = 2 * 64 * dim * 4
            with mock.patch.object(Align_GPU_Kernels, "_UPLOAD_CHUNK_BYTES", small_chunk_bytes), \
                    mock.patch.object(store, "get", side_effect=failing_get), \
                    mock.patch.object(torch.cuda, "Event", SpyEvent), \
                    mock.patch.object(torch.cuda, "device", SpyDevice):
                with self.assertRaisesRegex(RuntimeError, "simulated HDF5 read failure"):
                    Align_GPU_Kernels._upload_and_normalize_indices(
                        [0, 1, 2], lengths, dim, store, None, dev, torch.float32
                    )

            self.assertEqual(entered_devices, [dev])
            self.assertGreaterEqual(len(recorded_streams), 1)
            self.assertTrue(all(s is not None for s in recorded_streams))
            self.assertGreaterEqual(len(synced_events), 1)

    @staticmethod
    def _make_global_and_pos_tie_matrix(rows, cols, global_gap):
        """Construct a float32 score matrix where M vs M.T differs in both g_len and l_len.

        - Top-left 2x2 gadget creates a global DP up == left > diag tie with differing
          path lengths (orig_up has length 3 vs orig_left length 4).
        - Two equal-scoring local peaks at (2, cols - 2) [len 1, score 6.0] and
          (rows - 2, 3) [len 2, score 6.0] satisfy r1 < r2 and c1 > c2, so canonical
          row-major order selects len 1 while naive transposed order selects len 2.
        """
        mat = np.full((rows, cols), -5.0, dtype=np.float32)
        mat[0, 1] = np.float32(2.0 * global_gap)
        mat[2, cols - 2] = np.float32(8.0)
        mat[rows - 3, 2] = np.float32(5.0)
        mat[rows - 2, 3] = np.float32(5.0)
        return mat

    @staticmethod
    def _make_local_up_left_tie_matrix(rows, cols, local_gap):
        """Construct a float32 score matrix where local DP hits up == left > max(0, diag).

        Path A reaches (r0+1, c0+2) with score 10 + 3*local_gap in 3 steps; Path B reaches
        (r0+2, c0+1) with the same score in 4 steps. At (r0+2, c0+2) both gap transitions
        tie and beat diag, and a strong match at (r0+3, c0+3) carries the winner into
        max_local_length (5 canonical vs 6 transposed).
        """
        mat = np.full((rows, cols), -5.0, dtype=np.float32)
        r0 = 2
        c0 = min(cols - 5, 130 if cols > 128 else 2)
        mat[r0, c0] = np.float32(12.0)
        mat[r0 + 1, c0 + 2] = np.float32(2.0 + 2.0 * local_gap)
        mat[r0 + 3, c0 + 3] = np.float32(22.0)
        return mat

    @staticmethod
    def _run_dp_kernel_batch(kernels, matrices, transposed_flags, global_gap, local_gap, use_f64_gap):
        """Launch align_dp_f32_k4 or align_dp_f64gap_k4 directly on arbitrary float32 matrices."""
        dev = kernels.device
        num_p = len(matrices)
        max_c = max(int(m.shape[1]) for m in matrices)
        r_lens = [int(m.shape[0]) for m in matrices]
        c_lens = [int(m.shape[1]) for m in matrices]
        r_offs = np.cumsum([0] + r_lens[:-1], dtype=np.int32)
        total_r = int(sum(r_lens))
        total_c = int(sum(c_lens))

        packed_mat = np.zeros((total_r, max_c), dtype=np.float32)
        for off, m in zip(r_offs.tolist(), matrices):
            packed_mat[off:off + m.shape[0], :m.shape[1]] = m

        d_mat = torch.from_numpy(packed_mat).to(dev)
        d_r_off = torch.from_numpy(r_offs).to(dev)
        d_r_len = torch.tensor(r_lens, dtype=torch.int32, device=dev)
        d_c_off = torch.zeros(num_p, dtype=torch.int32, device=dev)
        d_c_len = torch.tensor(c_lens, dtype=torch.int32, device=dev)
        d_rs_off = torch.from_numpy(r_offs).to(dev)
        cs_offs = np.cumsum([0] + c_lens[:-1], dtype=np.int32)
        d_cs_off = torch.from_numpy(cs_offs).to(dev)
        d_trans = torch.tensor(transposed_flags, dtype=torch.int32, device=dev)

        # rm=0, rhis=0.5, cm=0, chis=0.5 makes cell == sim == packed_mat[r, c] bit-exact.
        d_rm = torch.zeros(total_r, dtype=torch.float32, device=dev)
        d_rs = torch.full((total_r,), 0.5, dtype=torch.float32, device=dev)
        d_cm = torch.zeros(total_c, dtype=torch.float32, device=dev)
        d_cs = torch.full((total_c,), 0.5, dtype=torch.float32, device=dev)

        sc_sz = total_r if max_c > 128 else 1
        sc_g = torch.empty(sc_sz, dtype=torch.float32, device=dev)
        sc_gl = torch.empty(sc_sz, dtype=torch.int32, device=dev)
        sc_l = torch.empty(sc_sz, dtype=torch.float32, device=dev)
        sc_ll = torch.empty(sc_sz, dtype=torch.int32, device=dev)
        out_gs = torch.empty(num_p, dtype=torch.float32, device=dev)
        out_gl = torch.empty(num_p, dtype=torch.int32, device=dev)
        out_ls = torch.empty(
            num_p, dtype=torch.float64 if use_f64_gap else torch.float32, device=dev
        )
        out_ll = torch.empty(num_p, dtype=torch.int32, device=dev)

        dp_name = "align_dp_f64gap_k4" if use_f64_gap else "align_dp_f32_k4"
        kernels.launch(
            dp_name,
            ((num_p + 3) // 4, 1, 1),
            (128, 1, 1),
            [
                d_mat, max_c, d_r_off, d_r_len, d_c_off, d_c_len, d_rs_off, d_cs_off, d_trans,
                d_rm, d_rs, d_cm, d_cs, sc_g, sc_gl, sc_l, sc_ll,
                out_gs, out_gl, out_ls, out_ll, num_p,
                float(global_gap), float(local_gap), 2.0,
            ],
        )
        torch.cuda.synchronize(dev)
        return (
            out_gs.cpu().numpy(),
            out_gl.cpu().numpy().view(np.uint32),
            out_ls.cpu().numpy(),
            out_ll.cpu().numpy().view(np.uint32),
        )

    def test_transposed_dp_tie_breaking_matches_cpu_cuda_and_hip(self):
        # M1: Orientation-canonical tie-breaking must produce identical (g_score, g_len,
        # l_score, l_len) for M (transposed=0) and M.T (transposed=1) across:
        #   - align_dp_f32_k4 and align_dp_f64gap_k4
        #   - C <= 128 single-swath, C > 128 multi-swath, and asymmetric cross-swath shapes
        #   - global up==left>diag, local up==left>max(0,diag), and local best_pos ties
        #   - both CUDA warp-shuffle and AMD HIP shared-memory lockstep kernel paths.
        dev = torch.device("cuda:0")
        cuda_kernels = Align_GPU_Kernels.align_kernels(dev)
        self.assertIsNotNone(cuda_kernels)
        hip_source = "#define __HIP_PLATFORM_AMD__ 1\n" + Align_GPU_Kernels.KERNEL_SOURCE
        with mock.patch.object(Align_GPU_Kernels, "KERNEL_SOURCE", hip_source):
            hip_kernels = Align_GPU_Kernels.AlignKernels(dev, read_cache=False)

        shapes = [(12, 20), (20, 12), (30, 150), (150, 30), (140, 160), (160, 140)]
        gap_configs = [
            (0.0, -2.0, False, True),
            (-0.5, -1.5, True, True),
            (-0.7, -1.3, True, False),
        ]

        for backend_label, kernels in (("cuda", cuda_kernels), ("hip", hip_kernels)):
            for global_gap, local_gap, use_f64, exact_dyadic in gap_configs:
                batch_mats = []
                batch_flags = []
                batch_meta = []

                for rows, cols in shapes:
                    m_gp = self._make_global_and_pos_tie_matrix(rows, cols, global_gap)
                    ref_gp = global_local_scores(m_gp, global_gap, local_gap)
                    ref_gp_t = global_local_scores(np.ascontiguousarray(m_gp.T), global_gap, local_gap)
                    if exact_dyadic:
                        # Prove the test matrix genuinely breaks naive transpose on both lengths.
                        self.assertNotEqual(
                            int(ref_gp[1]), int(ref_gp_t[1]),
                            f"Expected g_len transpose diff for {(rows, cols)} gaps={(global_gap, local_gap)}",
                        )
                    self.assertNotEqual(
                        int(ref_gp[3]), int(ref_gp_t[3]),
                        f"Expected l_len best_pos transpose diff for {(rows, cols)} gaps={(global_gap, local_gap)}",
                    )
                    # Include canonical (M, tr=0), fixed transposed (M.T, tr=1), and unfixed (M.T, tr=0)
                    # in the SAME multi-warp batch to exercise mixed orientations per block.
                    batch_mats.extend([m_gp, np.ascontiguousarray(m_gp.T), np.ascontiguousarray(m_gp.T)])
                    batch_flags.extend([0, 1, 0])
                    batch_meta.extend([
                        ("gp_canon", (rows, cols), ref_gp),
                        ("gp_trans1", (rows, cols), ref_gp),
                        ("gp_trans0_control", (rows, cols), ref_gp_t),
                    ])

                    m_lu = self._make_local_up_left_tie_matrix(rows, cols, local_gap)
                    ref_lu = global_local_scores(m_lu, global_gap, local_gap)
                    ref_lu_t = global_local_scores(np.ascontiguousarray(m_lu.T), global_gap, local_gap)
                    if exact_dyadic:
                        self.assertNotEqual(
                            int(ref_lu[3]), int(ref_lu_t[3]),
                            f"Expected local up/left transpose diff for {(rows, cols)} gaps={(global_gap, local_gap)}",
                        )
                    batch_mats.extend([m_lu, np.ascontiguousarray(m_lu.T), np.ascontiguousarray(m_lu.T)])
                    batch_flags.extend([0, 1, 0])
                    batch_meta.extend([
                        ("lu_canon", (rows, cols), ref_lu),
                        ("lu_trans1", (rows, cols), ref_lu),
                        ("lu_trans0_control", (rows, cols), ref_lu_t),
                    ])

                gs, gl, ls, ll = self._run_dp_kernel_batch(
                    kernels, batch_mats, batch_flags, global_gap, local_gap, use_f64
                )
                for idx, (tag, shape, ref) in enumerate(batch_meta):
                    rgs, rgl, rls, rll = ref
                    ctx = f"{backend_label} {tag} shape={shape} gaps={(global_gap, local_gap)}"
                    self.assertEqual(int(gl[idx]), int(rgl), f"g_len mismatch: {ctx}")
                    self.assertEqual(int(ll[idx]), int(rll), f"l_len mismatch: {ctx}")
                    if exact_dyadic:
                        self.assertEqual(float(gs[idx]), float(rgs), f"g_score mismatch: {ctx}")
                        self.assertEqual(float(ls[idx]), float(rls), f"l_score mismatch: {ctx}")
                    else:
                        self.assertLess(
                            abs(float(gs[idx]) - float(rgs)) / max(1, int(rgl)), 1e-5, ctx
                        )
                        self.assertLess(
                            abs(float(ls[idx]) - float(rls)) / max(1, int(rll)), 1e-5, ctx
                        )

    def test_transposed_dense_and_sparse_pipelines_match_cpu_on_tied_embeddings(self):
        # M1 end-to-end: Force dynamic transposition in BOTH dense blk_pairs (short block
        # paired against long block) AND sparse slabs (short low-degree si paired against
        # longer/higher-degree sj) on homopolymer and repeated-block embeddings, covering
        # C <= 128 and C > 128 and both f32-exact and f64-gap kernels, plus _score_pairs_direct.
        dev = torch.device("cuda:0")
        kernels = Align_GPU_Kernels.align_kernels(dev)
        self.assertIsNotNone(kernels)

        dim = 16
        c_mid = float(1.0 + np.log(0.5 * (1.0 + np.exp(-1.0))))
        s_mid = float(np.sqrt(1.0 - 2.0 * c_mid * c_mid))
        basis = np.zeros((4, dim), dtype=np.float32)
        basis[0, 0] = 1.0
        basis[1, 1] = 1.0
        basis[2, 0] = c_mid
        basis[2, 1] = c_mid
        basis[2, 2] = s_mid
        basis[3, 3] = 1.0

        def make_cmid_pair(la, lb):
            ha = (la - 2) // 2
            hb = (lb - 2) // 2
            ta = [0, 3] + [0] * (ha - 1) + [1] * ha + [2]
            tb = [1, 2] + [1] * (hb - 2) + [0] * hb + [2, 3]
            return basis[ta].copy(), basis[tb].copy()

        def make_repeated_block_emb(length, block_pattern):
            arr = np.zeros((length, dim), dtype=np.float32)
            for idx in range(length):
                arr[idx, block_pattern[idx % len(block_pattern)]] = 1.0
            return arr

        # 1. Verify _score_pairs_direct directly with transposed=1 vs transposed=0 on a tied
        #    short embedding pair where untransposed M.T diverges in g_len.
        ea_tie, eb_tie = make_cmid_pair(16, 32)
        with torch.cuda.device(dev), torch.inference_mode():
            ta_g = torch.nn.functional.normalize(torch.from_numpy(ea_tie).to(dev), p=2, dim=-1)
            tb_g = torch.nn.functional.normalize(torch.from_numpy(eb_tie).to(dev), p=2, dim=-1)
            z1 = torch.zeros(1, dtype=torch.int32, device=dev)
            r32 = torch.tensor([32], dtype=torch.int32, device=dev)
            c16 = torch.tensor([16], dtype=torch.int32, device=dev)
            r16 = torch.tensor([16], dtype=torch.int32, device=dev)
            c32 = torch.tensor([32], dtype=torch.int32, device=dev)

            gs_c, gl_c, ls_c, ll_c, nf_c = kernels._score_pairs_direct(
                torch.mm(ta_g, tb_g.T), z1, r16, z1, c32, z1, z1, 16, 32,
                0.0, -2.0, use_f64_gap=False, transposed=[0],
            )
            gs_t1, gl_t1, ls_t1, ll_t1, nf_t1 = kernels._score_pairs_direct(
                torch.mm(tb_g, ta_g.T), z1, r32, z1, c16, z1, z1, 32, 16,
                0.0, -2.0, use_f64_gap=False, transposed=[1],
            )
            _gs_t0, gl_t0, _ls_t0, _ll_t0, _nf_t0 = kernels._score_pairs_direct(
                torch.mm(tb_g, ta_g.T), z1, r32, z1, c16, z1, z1, 32, 16,
                0.0, -2.0, use_f64_gap=False, transposed=[0],
            )
            self.assertEqual((nf_c, nf_t1), (0, 0))
            # Untransposed M.T disagrees with canonical M, while transposed=1 restores exact parity.
            self.assertNotEqual(int(gl_t0[0]), int(gl_c[0]))
            self.assertEqual(int(gl_t1[0]), int(gl_c[0]))
            self.assertEqual(int(ll_t1[0]), int(ll_c[0]))
            self.assertAlmostEqual(float(gs_t1[0]), float(gs_c[0]), places=4)
            self.assertAlmostEqual(float(ls_t1[0]), float(ls_c[0]), places=4)

        # 2. Build a multi-block embedding store containing:
        #    - Block 0 (64 short seqs of length 16 -> 1024 rows = 1 full block under
        #      block_res=1024, blk_maxlen=16): includes ea_tie (idx 0) and repeated-block
        #      embeddings over basis {0, 1, 2, 3}.
        #    - Block 1+ (longer seqs of lengths 32..220, blk_maxlen=220): includes eb_tie
        #      (idx 64, len 32) and multi-swath (>128) repeated-block embeddings over {0..3}.
        #    Pairing short si (block 0) with long sj (block 1+) forces blk_maxlen[bi] <
        #    blk_maxlen[bj] in dense mode AND lengths[si] < lengths[sj] in sparse mode,
        #    guaranteeing transposed == 1 on BOTH (ea_tie, eb_tie) and multi-swath (>128)
        #    pairs!
        short_patterns = [
            [0, 0, 1, 2, 1, 3],
            [0, 1, 1, 2, 0, 3],
            [1, 0, 2, 3, 1, 2],
            [2, 1, 0, 3, 2, 0],
        ]
        short_embs = [ea_tie] + [
            make_repeated_block_emb(16, short_patterns[i % len(short_patterns)])
            for i in range(63)
        ]  # 0..63: 64 * 16 = 1024 rows -> fills block 0 cleanly (blk_maxlen = 16)
        long_embs = [
            eb_tie,                                            # 64: len 32 (pairs with ea_tie!)
            make_repeated_block_emb(82, [0, 1, 2, 3, 1, 0]),   # 65: len 82 (sum=1024 -> exact block 0)
            make_repeated_block_emb(150, [0, 1, 2, 3, 0, 1]),  # 66: len 150 (>128)
            make_repeated_block_emb(160, [0, 1, 2, 2, 3, 1]),  # 67: len 160 (>128)
            make_repeated_block_emb(180, [1, 0, 0, 2, 3, 1]),  # 68: len 180 (>128)
            make_repeated_block_emb(200, [0, 2, 1, 1, 3, 0]),  # 69: len 200 (>128)
            make_repeated_block_emb(220, [2, 1, 0, 3, 0, 1]),  # 70: len 220 (>128)
        ]
        # Dedicated homopolymer pair (exact cos=1.0 -> sim=1.0 -> cell=0.0 everywhere,
        # short 16 vs long 150) tested in sparse mode as well.
        homo_short_idx = len(short_embs) + len(long_embs)
        homo_long_idx = homo_short_idx + 1
        all_embs = short_embs + long_embs + [
            np.ones((16, dim), dtype=np.float32),
            np.ones((150, dim), dtype=np.float32),
        ]
        lengths = [int(e.shape[0]) for e in all_embs]
        headers = [f"tied_seq_{i}" for i in range(len(all_embs))]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "tied_embs.h5")
            with h5py.File(path, "w") as hf:
                grp = hf.create_group("embeddings")
                for header, emb in zip(headers, all_embs):
                    grp.create_dataset(header, data=emb)
            store = engine.EmbeddingTileStore(path, headers, host_cache_setting="auto")

            short_ids = list(range(len(short_embs)))
            long_ids = list(range(len(short_embs), len(short_embs) + len(long_embs)))

            # Dense bipartite workload: every short sequence in block 0 paired with every
            # long sequence in block 1+ (cross-block density = 1.0 >= 0.20 -> dense blk_pairs
            # with transposed == 1, including the tie-sensitive (0, 64) pair).
            dense_tasks = [
                (si, sj, headers[si], headers[sj])
                for si in short_ids
                for sj in long_ids
            ]
            # Sparse workload: low-density (<20%) subset pairing short si with longer sj
            # (every sequence has degree 1 so lengths[si] < lengths[sj] forces sj as row
            # and si as col -> transposed == 1 on every pair, including (0, 64), >128, and homopolymer).
            sparse_tasks = [
                (0, 64, headers[0], headers[64]),
                (5, 66, headers[5], headers[66]),
                (1, 67, headers[1], headers[67]),
                (2, 68, headers[2], headers[68]),
                (3, 69, headers[3], headers[69]),
                (4, 70, headers[4], headers[70]),
                (homo_short_idx, homo_long_idx, headers[homo_short_idx], headers[homo_long_idx]),
            ]

            def cpu_reference_for_tasks(task_list, g_gap, l_gap):
                ref_map = {}
                for si, sj, _, _ in task_list:
                    ti = torch.nn.functional.normalize(
                        torch.from_numpy(all_embs[si]).double(), p=2, dim=-1
                    )
                    tj = torch.nn.functional.normalize(
                        torch.from_numpy(all_embs[sj]).double(), p=2, dim=-1
                    )
                    cos = torch.mm(ti, tj.T).clamp(-1.0, 1.0)
                    sim = torch.exp(-(1.0 - cos))
                    rm = sim.mean(dim=1, keepdim=True)
                    rs = sim.std(dim=1, keepdim=True, correction=0)
                    cm = sim.mean(dim=0, keepdim=True)
                    cs = sim.std(dim=0, keepdim=True, correction=0)
                    sc = ((((sim - rm) / (rs + 1e-8)) + ((sim - cm) / (cs + 1e-8))) / 2.0).float().numpy()
                    rgs, rgl, rls, rll = global_local_scores(sc, g_gap, l_gap)
                    ref_map[(si, sj)] = (float(rls), int(rll), float(rgs), int(rgl))
                return ref_map

            for mode_name, task_list in (("dense", dense_tasks), ("sparse", sparse_tasks)):
                for global_gap, local_gap, expected_kernel in (
                    (0.0, -2.0, "align_dp_f32_k4"),
                    (-0.5, -1.5, "align_dp_f64gap_k4"),
                    (-0.7, -1.3, "align_dp_f64gap_k4"),
                ):
                    seen_transposed_flags = []
                    real_launch = kernels.launch

                    def spy_launch(name, grid, block, args, stream=None):
                        if name == expected_kernel:
                            # Synchronize before reading device metadata arg[8] (pair_transposed).
                            torch.cuda.synchronize(dev)
                            seen_transposed_flags.extend(args[8].cpu().tolist())
                        return real_launch(name, grid, block, args, stream=stream)

                    with mock.patch.object(kernels, "launch", side_effect=spy_launch):
                        fused = Align_GPU_Kernels.run_fused_gpu_alignment_pipeline(
                            task_list,
                            store=store,
                            lengths=lengths,
                            device=dev,
                            lanes=2,
                            global_gap=global_gap,
                            local_gap=local_gap,
                            precision="float32",
                            matrix_budget_override=4 * 1024 * 1024,  # block_res=1024 -> multi-block
                            cuda_matmul_precision_ctx=engine.cuda_matmul_precision,
                            precision_compute_dtype_fn=engine.precision_compute_dtype,
                            precision_element_bytes_fn=engine.precision_element_bytes,
                            accelerator_memory_plan_fn=engine.accelerator_memory_plan,
                            partition_tiles_fn=engine._partition_tiles,
                        )

                    self.assertEqual(len(seen_transposed_flags), len(task_list))
                    self.assertTrue(
                        all(flag == 1 for flag in seen_transposed_flags),
                        f"Expected all pairs dynamically transposed (transposed=1) in "
                        f"{mode_name} {expected_kernel}, got {seen_transposed_flags[:16]}",
                    )
                    ref_map = cpu_reference_for_tasks(task_list, global_gap, local_gap)
                    self.assertEqual(len(fused), len(task_list))
                    for si, sj, l_s, l_l, g_s, g_l in fused:
                        rls, rll, rgs, rgl = ref_map[(si, sj)]
                        ctx = f"{mode_name} {expected_kernel} pair=({si},{sj}) gaps={(global_gap, local_gap)}"
                        self.assertEqual(int(g_l), rgl, f"g_len mismatch: {ctx}")
                        self.assertEqual(int(l_l), rll, f"l_len mismatch: {ctx}")
                        self.assertLess(
                            abs(float(g_s) - rgs) / max(1, rgl), 1e-4, f"g_score drift: {ctx}"
                        )
                        self.assertLess(
                            abs(float(l_s) - rls) / max(1, rll), 1e-4, f"l_score drift: {ctx}"
                        )


if __name__ == "__main__":
    unittest.main()
