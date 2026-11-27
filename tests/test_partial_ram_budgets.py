# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quota-aware host memory detection and partial-RAM window budget planning."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import h5py
import numpy as np
import torch

import Embedding_Alignment_Engine as engine
from tests.embedding_fixtures import write_embedding_arrays
from tools import Align_Similarity_Matrix as align_tool
from utilities import Prefilter_Calibration


def _make_h5(temp_dir, lengths, dim=64, dtype=np.float16, seed=42):
    rng = np.random.default_rng(seed)
    path = os.path.join(temp_dir, "embs.h5")
    headers = [f"s{i:03d}" for i in range(len(lengths))]
    write_embedding_arrays(
        path,
        {h: rng.standard_normal((L, dim)).astype(dtype) for h, L in zip(headers, lengths)},
    )
    return path, headers


def _store_with_budget(path, headers, budget_bytes):
    with mock.patch.object(engine, "resolve_host_cache_bytes", return_value=int(budget_bytes)):
        return engine.EmbeddingTileStore(path, headers, host_cache_setting="auto")


class SystemMemoryQuotaTests(unittest.TestCase):
    """cgroup v1/v2, SLURM memory/CPU parsing, system_memory_bytes, and resolve_host_cache_bytes."""

    def test_system_memory_respects_cgroup_v2_v1_and_slurm_quotas(self):
        gib = engine.GIB
        host_mem = mock.Mock(total=512 * gib, available=400 * gib)

        with tempfile.TemporaryDirectory() as tmp:
            # 1) Cgroup v2: parent limits to 32 GiB, leaf has 'max', current=6 GiB with 2 GiB inactive_file -> 28 GiB avail
            cg_root = os.path.join(tmp, "cg2")
            job_dir = os.path.join(cg_root, "slurm", "job_42")
            os.makedirs(job_dir)
            with open(os.path.join(cg_root, "slurm", "memory.max"), "w") as f:
                f.write(str(32 * gib))
            with open(os.path.join(cg_root, "slurm", "memory.current"), "w") as f:
                f.write(str(6 * gib))
            with open(os.path.join(cg_root, "slurm", "memory.stat"), "w") as f:
                f.write(f"anon {4 * gib}\ninactive_file {2 * gib}\n")
            with open(os.path.join(job_dir, "memory.max"), "w") as f:
                f.write("max\n")
            proc_cg2 = os.path.join(tmp, "proc_cg2")
            with open(proc_cg2, "w") as f:
                f.write("0::/slurm/job_42\n")

            v2 = engine._cgroup_v2_limits(
                512 * gib, proc_cgroup=proc_cg2, cgroup_root=cg_root
            )
            self.assertEqual(v2, (32 * gib, 28 * gib))

            # 2) Cgroup v1: 24 GiB limit, 5 GiB usage with 1 GiB total_inactive_file -> 20 GiB avail
            cg1_root = os.path.join(tmp, "cg1")
            cg1_job = os.path.join(cg1_root, "docker", "abc")
            os.makedirs(cg1_job)
            with open(os.path.join(cg1_job, "memory.limit_in_bytes"), "w") as f:
                f.write(str(24 * gib))
            with open(os.path.join(cg1_job, "memory.usage_in_bytes"), "w") as f:
                f.write(str(5 * gib))
            with open(os.path.join(cg1_job, "memory.stat"), "w") as f:
                f.write(f"total_inactive_file {1 * gib}\n")
            proc_cg1 = os.path.join(tmp, "proc_cg1")
            with open(proc_cg1, "w") as f:
                f.write("4:memory:/docker/abc\n")

            v1 = engine._cgroup_v1_limits(
                512 * gib, proc_cgroup=proc_cg1, cgroup_root=cg1_root
            )
            self.assertEqual(v1, (24 * gib, 20 * gib))

            # 3) SLURM env vars: SLURM_MEM_PER_NODE=32768 (32 GiB), RSS=4 GiB -> 28 GiB avail -> auto budget=20 GiB
            fake_proc = mock.Mock()
            fake_proc.memory_info.return_value = mock.Mock(rss=4 * gib)
            with mock.patch("psutil.virtual_memory", return_value=host_mem), \
                 mock.patch("psutil.Process", return_value=fake_proc), \
                 mock.patch.object(engine, "_cgroup_v2_limits", return_value=None), \
                 mock.patch.object(engine, "_cgroup_v1_limits", return_value=None), \
                 mock.patch.dict(os.environ, {"SLURM_MEM_PER_NODE": "32768"}, clear=False):
                self.assertEqual(engine.system_memory_bytes(), (32 * gib, 28 * gib))
                self.assertEqual(engine.resolve_host_cache_bytes("auto"), 20 * gib)

            # 4) SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK (4096 MiB * 4 = 16 GiB)
            with mock.patch("psutil.Process", return_value=fake_proc):
                sl = engine._slurm_memory_limits(
                    512 * gib,
                    environ={"SLURM_MEM_PER_CPU": "4096", "SLURM_CPUS_PER_TASK": "4"},
                )
            self.assertEqual(sl, (16 * gib, 12 * gib))

    def test_slurm_memory_parsing_suffixes_overflow_and_cpu_lists(self):
        """Medium #5: _parse_slurm_mem_bytes and _slurm_memory_limits handle overflow, suffixes, and CPU lists."""
        gib = 1024 ** 3
        mib = 1024 ** 2
        kib = 1024
        tib = 1024 ** 4
        self.assertIsNone(engine._parse_slurm_mem_bytes("1e308T"))
        self.assertIsNone(engine._parse_slurm_mem_bytes("0"))
        self.assertIsNone(engine._parse_slurm_mem_bytes("0GB"))
        self.assertIsNone(engine._parse_slurm_mem_bytes("-8G"))
        self.assertIsNone(engine._parse_slurm_mem_bytes("inf"))
        self.assertIsNone(engine._parse_slurm_mem_bytes("nan"))
        self.assertEqual(engine._parse_slurm_mem_bytes("16GB"), 16 * gib)
        self.assertEqual(engine._parse_slurm_mem_bytes("8GiB"), 8 * gib)
        self.assertEqual(engine._parse_slurm_mem_bytes("8gib"), 8 * gib)
        self.assertEqual(engine._parse_slurm_mem_bytes("512MB"), 512 * mib)
        self.assertEqual(engine._parse_slurm_mem_bytes("256MiB"), 256 * mib)
        self.assertEqual(engine._parse_slurm_mem_bytes("1024KB"), 1024 * kib)
        self.assertEqual(engine._parse_slurm_mem_bytes("2KiB"), 2 * kib)
        self.assertEqual(engine._parse_slurm_mem_bytes("1.5TB"), int(1.5 * tib))
        self.assertEqual(engine._parse_slurm_mem_bytes("2TiB"), 2 * tib)
        self.assertEqual(engine._parse_slurm_mem_bytes("4096"), 4096 * mib)

        fake_proc = mock.Mock()
        fake_proc.memory_info.return_value = mock.Mock(rss=2 * gib)
        with mock.patch("psutil.Process", return_value=fake_proc):
            # SLURM_CPUS_ON_NODE="4,4" -> 4 CPUs * 4 GiB = 16 GiB (avail 14 GiB)
            sl_comma = engine._slurm_memory_limits(
                512 * gib,
                environ={"SLURM_MEM_PER_CPU": "4GB", "SLURM_CPUS_ON_NODE": "4,4"},
            )
            self.assertEqual(sl_comma, (16 * gib, 14 * gib))
            # SLURM_CPUS_ON_NODE="4(x2)" -> 4 CPUs * 4 GiB = 16 GiB (avail 14 GiB)
            sl_paren = engine._slurm_memory_limits(
                512 * gib,
                environ={"SLURM_MEM_PER_CPU": "4GiB", "SLURM_CPUS_ON_NODE": "4(x2)"},
            )
            self.assertEqual(sl_paren, (16 * gib, 14 * gib))

    def test_resolve_host_cache_bytes_clamps_explicit_when_safe_limit_is_zero(self):
        """HIGH #3: explicit HOST_CACHE_GB is clamped to 0 when total>0 and available<=reserve."""
        gib = engine.GIB

        # 1) 8 GiB Docker/SLURM container: reserve = max(8 GiB, 25%) = 8 GiB -> safe_limit = 0.
        with mock.patch.object(engine, "system_memory_bytes", return_value=(8 * gib, 6 * gib)):
            self.assertEqual(engine.resolve_host_cache_bytes("auto"), 0)
            self.assertEqual(engine.resolve_host_cache_bytes(16), 0)
            self.assertEqual(engine.resolve_host_cache_bytes(4), 0)
            self.assertEqual(engine.resolve_host_cache_bytes(0), 0)

        # 2) Larger host under memory pressure: total=32 GiB, available=8 GiB == reserve -> safe_limit = 0.
        with mock.patch.object(engine, "system_memory_bytes", return_value=(32 * gib, 8 * gib)):
            self.assertEqual(engine.resolve_host_cache_bytes("auto"), 0)
            self.assertEqual(engine.resolve_host_cache_bytes(16), 0)

        # 3) total>0 with zero available memory -> safe_limit = 0, explicit still clamped to 0.
        with mock.patch.object(engine, "system_memory_bytes", return_value=(16 * gib, 0)):
            self.assertEqual(engine.resolve_host_cache_bytes("auto"), 0)
            self.assertEqual(engine.resolve_host_cache_bytes(8), 0)

        # 4) System memory unreadable (0, 0): 'auto' returns 0, explicit setting passes through.
        with mock.patch.object(engine, "system_memory_bytes", return_value=(0, 0)):
            self.assertEqual(engine.resolve_host_cache_bytes("auto"), 0)
            self.assertEqual(engine.resolve_host_cache_bytes(16), 16 * gib)

        # 5) Normal headroom: explicit request capped at safe_limit (20 - 16 = 4 GiB).
        with mock.patch.object(engine, "system_memory_bytes", return_value=(64 * gib, 20 * gib)):
            self.assertEqual(engine.resolve_host_cache_bytes("auto"), 4 * gib)
            self.assertEqual(engine.resolve_host_cache_bytes(16), 4 * gib)
            self.assertEqual(engine.resolve_host_cache_bytes(2), 2 * gib)

    def test_cgroup_v1_hierarchical_memory_limit(self):
        """MEDIUM #3: cgroup v1 reads hierarchical_memory_limit when limit_in_bytes is unlimited or looser."""
        gib = engine.GIB
        host_total = 512 * gib
        unlimited_v1 = (1 << 63) - 4096

        def run_case(tmp, name, limit_raw, usage_raw, stat_text):
            root = os.path.join(tmp, name)
            job_dir = os.path.join(root, "slurm", "job_9")
            os.makedirs(job_dir)
            if limit_raw is not None:
                with open(os.path.join(job_dir, "memory.limit_in_bytes"), "w") as f:
                    f.write(str(limit_raw) + "\n")
            if usage_raw is not None:
                with open(os.path.join(job_dir, "memory.usage_in_bytes"), "w") as f:
                    f.write(str(usage_raw) + "\n")
            if stat_text is not None:
                with open(os.path.join(job_dir, "memory.stat"), "w") as f:
                    f.write(stat_text)
            proc_cg = os.path.join(tmp, f"proc_{name}")
            with open(proc_cg, "w") as f:
                f.write("4:memory:/slurm/job_9\n")
            return engine._cgroup_v1_limits(
                host_total, proc_cgroup=proc_cg, cgroup_root=root
            )

        with tempfile.TemporaryDirectory() as tmp:
            # 1) limit_in_bytes is unlimited sentinel (~2^63) -> falls back to hierarchical_memory_limit.
            res1 = run_case(
                tmp,
                "sentinel",
                unlimited_v1,
                5 * gib,
                f"hierarchical_memory_limit {16 * gib}\ntotal_inactive_file {1 * gib}\n",
            )
            self.assertEqual(res1, (16 * gib, 12 * gib))

            # 2) limit_in_bytes >= host_total -> falls back to hierarchical_memory_limit with inactive_file.
            res2 = run_case(
                tmp,
                "ge_host",
                host_total,
                6 * gib,
                f"hierarchical_memory_limit {20 * gib}\ninactive_file {2 * gib}\n",
            )
            self.assertEqual(res2, (20 * gib, 16 * gib))

            # 3) Both valid, hierarchical_memory_limit tighter -> min(limit, h_limit).
            res3 = run_case(
                tmp,
                "both_h_tighter",
                32 * gib,
                4 * gib,
                f"hierarchical_memory_limit {12 * gib}\ntotal_inactive_file {1 * gib}\n",
            )
            self.assertEqual(res3, (12 * gib, 9 * gib))

            # 4) Both valid, limit_in_bytes tighter -> min(limit, h_limit).
            res4 = run_case(
                tmp,
                "both_l_tighter",
                12 * gib,
                4 * gib,
                f"hierarchical_memory_limit {32 * gib}\n",
            )
            self.assertEqual(res4, (12 * gib, 8 * gib))

            # 5) Both unlimited -> None.
            res5 = run_case(
                tmp,
                "both_unlimited",
                unlimited_v1,
                2 * gib,
                f"hierarchical_memory_limit {unlimited_v1}\n",
            )
            self.assertIsNone(res5)

            # 6) limit_in_bytes unlimited, hierarchical_memory_limit invalid (0 / >= host / non-int) -> None.
            self.assertIsNone(
                run_case(tmp, "h_zero", unlimited_v1, 2 * gib, "hierarchical_memory_limit 0\n")
            )
            self.assertIsNone(
                run_case(tmp, "h_ge_host", unlimited_v1, 2 * gib, f"hierarchical_memory_limit {host_total}\n")
            )
            self.assertIsNone(
                run_case(tmp, "h_bad", unlimited_v1, 2 * gib, "hierarchical_memory_limit notanint\n")
            )

            # 7) limit_in_bytes absent, valid hierarchical_memory_limit, no usage file -> (h_limit, h_limit).
            res7 = run_case(
                tmp,
                "missing_limit_and_usage",
                None,
                None,
                f"hierarchical_memory_limit {10 * gib}\n",
            )
            self.assertEqual(res7, (10 * gib, 10 * gib))


class WindowBudgetPlanningTests(unittest.TestCase):
    """plan_windowed_budgets boundaries, full-cache OOM fallback, VRAM skip, and Apple MPS exclusion."""

    def test_plan_windowed_budgets_boundaries(self):
        lengths = [100] * 20
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = _make_h5(tmp, lengths, dim=64, dtype=np.float16)
            # Each seq = 100 * 64 * 2 = 12,800 B; total = 256,000 B
            # 1) Full-fit -> fully_cached=True, plan returns None
            s_full = _store_with_budget(path, headers, 512_000)
            self.assertTrue(s_full.fully_cached)
            self.assertIsNone(engine.plan_windowed_budgets(s_full))

            # 2) Zero budget -> None
            s_zero = _store_with_budget(path, headers, 0)
            self.assertFalse(s_zero.fully_cached)
            self.assertIsNone(engine.plan_windowed_budgets(s_zero))

            # 3) Too small (< 3 * max_seq = 38,400 B) -> None
            s_tiny = _store_with_budget(path, headers, 30_000)
            self.assertIsNone(engine.plan_windowed_budgets(s_tiny))

            # 4) Valid partial budget (40% of total = 102,400 B) -> anchor + 2*slab <= budget
            s_part = _store_with_budget(path, headers, 102_400)
            budgets = engine.plan_windowed_budgets(s_part)
            self.assertIsNotNone(budgets)
            anchor_b, slab_b = budgets
            self.assertGreaterEqual(anchor_b, 12_800)
            self.assertGreaterEqual(slab_b, 12_800)
            self.assertLessEqual(anchor_b + 2 * slab_b, 102_400)

    def test_full_cache_memory_error_falls_back_cleanly(self):
        lengths = [64] * 8
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = _make_h5(tmp, lengths, dim=32, dtype=np.float16)
            orig_empty = np.empty

            def fail_large(shape, *args, **kwargs):
                if isinstance(shape, tuple) and len(shape) == 2 and shape[0] == sum(lengths):
                    raise MemoryError("simulated host OOM")
                return orig_empty(shape, *args, **kwargs)

            with mock.patch.object(engine, "resolve_host_cache_bytes", return_value=10 * 1024 * 1024), \
                 mock.patch.object(engine.np, "empty", side_effect=fail_large):
                store = engine.EmbeddingTileStore(path, headers, host_cache_setting="auto")
            self.assertFalse(store.fully_cached)
            self.assertEqual(store.cached_bytes, 0)
            arr = store.get(3)
            self.assertEqual(arr.shape, (64, 32))

    def test_vram_single_batch_skip_and_prefilter_pair_keys_dense_parity(self):
        # 1) Single-batch VRAM-fit skips windowing; multi-batch still activates windowing.
        lengths = [100] * 20
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = _make_h5(tmp, lengths, dim=64, dtype=np.float16)
            # total source = 256,000 B; total fp32 = 512,000 B; budget = 102,400 B (40%)
            store = _store_with_budget(path, headers, 102_400)
            fake_plan = mock.Mock(tile_cache_bytes=1_000_000)  # holds entire 512 KB fp32 file
            with mock.patch.object(engine, "accelerator_memory_plan", return_value=fake_plan):
                # Single batch (num_tasks <= batch_size) -> skip windowing
                self.assertIsNone(
                    engine.plan_windowed_budgets(
                        store, device=torch.device("cuda:0"), num_tasks=50, batch_size=100
                    )
                )
                # Multi-batch (num_tasks > batch_size) -> activate windowing
                self.assertIsNotNone(
                    engine.plan_windowed_budgets(
                        store, device=torch.device("cuda:0"), num_tasks=250, batch_size=100
                    )
                )

            # 2) prefilter_pair_keys matches dense pair_similarity bit-for-bit below STREAMED_SIMILARITY_BYTES,
            #    and force_streamed=True matches within float32 ULP.
            with mock.patch.multiple(
                align_tool, FULL_INPUT_HDF5=path, PREFILTER_STRENGTH=60
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                keys_dense, cut_dense = align_tool.prefilter_pair_keys(headers, lengths)
                keys_stream, cut_stream = align_tool.prefilter_pair_keys(
                    headers, lengths, force_streamed=True
                )
            with h5py.File(path, "r") as hf:
                sim_dense = Prefilter_Calibration.pair_similarity(hf["embeddings"], headers, lengths)
            upper = np.triu(np.ones((len(headers), len(headers)), dtype=bool), k=1)
            want_threshold = float(np.percentile(sim_dense[upper], 60))
            rr, cc = np.nonzero((sim_dense >= want_threshold) & upper)
            want_keys = rr.astype(np.int64) * len(headers) + cc.astype(np.int64)
            self.assertEqual(cut_dense.threshold, want_threshold)
            np.testing.assert_array_equal(keys_dense, want_keys)
            self.assertAlmostEqual(cut_stream.threshold, want_threshold, places=6)
            self.assertGreater(keys_stream.size, 0)

    def test_plan_windowed_budgets_skips_mps_and_unified_memory(self):
        """HIGH #1: unified-memory MPS never allocates a host window alongside its GPU tile cache."""
        gib = engine.GIB
        lengths = [100] * 20
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = _make_h5(tmp, lengths, dim=64, dtype=np.float16)
            # Each seq = 12,800 B; total = 256,000 B; budget = 102,400 B (40%).
            store = _store_with_budget(path, headers, 102_400)
            self.assertIsNotNone(engine.plan_windowed_budgets(store))

            # 1) MPS device (torch.device and string forms) returns None immediately without calling planner.
            with mock.patch.object(
                engine,
                "accelerator_memory_plan",
                side_effect=AssertionError("MPS should skip window planning immediately"),
            ):
                self.assertIsNone(engine.plan_windowed_budgets(store, device=torch.device("mps")))
                self.assertIsNone(engine.plan_windowed_budgets(store, device="mps"))
                self.assertIsNone(engine.plan_windowed_budgets(store, device="mps:0"))

            # 2) Any device whose AcceleratorMemoryPlan reports unified_memory=True returns None.
            unified_plan = engine.AcceleratorMemoryPlan(
                free_bytes=16 * gib,
                total_bytes=16 * gib,
                usable_bytes=12 * gib,
                tile_cache_bytes=64_000,
                matrix_pool_bytes=64_000,
                matrix_bytes=32_000,
                reserve_bytes=4 * gib,
                lanes=2,
                inflight_slots=1,
                unified_memory=True,
            )
            with mock.patch.object(engine, "accelerator_memory_plan", return_value=unified_plan):
                self.assertIsNone(
                    engine.plan_windowed_budgets(
                        store, device=torch.device("cuda:0"), num_tasks=200, batch_size=50
                    )
                )

            # 3) accelerator_memory_plan propagates snapshot.unified_memory accurately.
            mps_plan = engine.accelerator_memory_plan(
                torch.device("mps"), memory_info=(8 * gib, 16 * gib)
            )
            cuda_plan = engine.accelerator_memory_plan(
                torch.device("cuda:0"), memory_info=(8 * gib, 16 * gib)
            )
            self.assertTrue(mps_plan.unified_memory)
            self.assertFalse(cuda_plan.unified_memory)

            # 4) Discrete CUDA plan (unified_memory=False) still activates windowing for multi-batch.
            discrete_plan = engine.AcceleratorMemoryPlan(
                free_bytes=16 * gib,
                total_bytes=16 * gib,
                usable_bytes=12 * gib,
                tile_cache_bytes=64_000,
                matrix_pool_bytes=64_000,
                matrix_bytes=32_000,
                reserve_bytes=4 * gib,
                lanes=2,
                inflight_slots=4,
                unified_memory=False,
            )
            with mock.patch.object(engine, "accelerator_memory_plan", return_value=discrete_plan):
                budgets = engine.plan_windowed_budgets(
                    store, device=torch.device("cuda:0"), num_tasks=200, batch_size=50
                )
            self.assertIsNotNone(budgets)
            self.assertLessEqual(budgets[0] + 2 * budgets[1], 102_400)


if __name__ == "__main__":
    unittest.main()
