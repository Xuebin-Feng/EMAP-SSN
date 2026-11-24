# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for runtime-compiled fused GPU alignment and SSEARCH kernels."""

import os
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
from utilities.Network_Kernels import global_local_scores
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
        self.assertIn(
            "#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC_RTC__) || defined(__HIP__)",
            Align_GPU_Kernels.KERNEL_SOURCE,
        )
        self.assertNotIn("#ifdef __HIP_PLATFORM_AMD__", Align_GPU_Kernels.KERNEL_SOURCE)
        self.assertIn(
            "float res = smem_warp[0];\n    __syncthreads();\n    return res;",
            Align_GPU_Kernels.KERNEL_SOURCE,
        )


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

    def test_disable_env_switch(self):
        with mock.patch.dict(os.environ, {Align_GPU_Kernels.DISABLE_ENVIRONMENT_VARIABLE: "0"}):
            self.assertTrue(Align_GPU_Kernels.kernels_disabled())
            self.assertIsNone(Align_GPU_Kernels.align_kernels(torch.device("cuda:0")))

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
                        # args[12..15] are sc_g, sc_gl, sc_l, sc_ll
                        seen_scratch_numels.append(int(args[12].numel()))
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
                    short_scratch_numels.append(int(args[12].numel()))
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


if __name__ == "__main__":
    unittest.main()
