# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for 2-level partial-RAM anchor+slab windowing across all 3 scenarios."""

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
from utilities import HDF5_Storage as Embedding_HDF5
from utilities import Prefilter_Calibration
from tools import Align_Similarity_Matrix as align_tool
from tools import Network_Injection as inject_tool

Hardware_Utils = align_tool.Hardware_Utils


class PartialRAMWindowTests(unittest.TestCase):
    def _make_h5(self, temp_dir, lengths, dim=64, dtype=np.float16, seed=42):
        rng = np.random.default_rng(seed)
        path = os.path.join(temp_dir, "embs.h5")
        headers = [f"s{i:03d}" for i in range(len(lengths))]
        with h5py.File(path, "w") as hf:
            grp = hf.create_group("embeddings")
            for h, L in zip(headers, lengths):
                grp.create_dataset(h, data=rng.standard_normal((L, dim)).astype(dtype))
        return path, headers

    def _store_with_budget(self, path, headers, budget_bytes):
        with mock.patch.object(engine, "resolve_host_cache_bytes", return_value=int(budget_bytes)):
            return engine.EmbeddingTileStore(path, headers, host_cache_setting="auto")

    def test_plan_windowed_budgets_boundaries(self):
        lengths = [100] * 20
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=64, dtype=np.float16)
            # Each seq = 100 * 64 * 2 = 12,800 B; total = 256,000 B
            # 1) Full-fit -> fully_cached=True, plan returns None
            s_full = self._store_with_budget(path, headers, 512_000)
            self.assertTrue(s_full.fully_cached)
            self.assertIsNone(engine.plan_windowed_budgets(s_full))

            # 2) Zero budget -> None
            s_zero = self._store_with_budget(path, headers, 0)
            self.assertFalse(s_zero.fully_cached)
            self.assertIsNone(engine.plan_windowed_budgets(s_zero))

            # 3) Too small (< 3 * max_seq = 38,400 B) -> None
            s_tiny = self._store_with_budget(path, headers, 30_000)
            self.assertIsNone(engine.plan_windowed_budgets(s_tiny))

            # 4) Valid partial budget (40% of total = 102,400 B) -> anchor + 2*slab <= budget
            s_part = self._store_with_budget(path, headers, 102_400)
            budgets = engine.plan_windowed_budgets(s_part)
            self.assertIsNotNone(budgets)
            anchor_b, slab_b = budgets
            self.assertGreaterEqual(anchor_b, 12_800)
            self.assertGreaterEqual(slab_b, 12_800)
            self.assertLessEqual(anchor_b + 2 * slab_b, 102_400)

    def test_full_cache_memory_error_falls_back_cleanly(self):
        lengths = [64] * 8
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=32, dtype=np.float16)
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

    def test_open_window_memory_error_falls_back_cleanly(self):
        lengths = [80] * 16
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=32, dtype=np.float16)
            # Each seq = 5,120 B; total = 81,920 B; budget = 32,000 B
            store = self._store_with_budget(path, headers, 32_000)
            anchor_b, slab_b = engine.plan_windowed_budgets(store)
            comp = np.zeros((16, 16), dtype=bool)
            with mock.patch.object(store, "open_window", return_value=False):
                with engine.WindowedPairSession.from_masks(
                    store, headers, comp, None,
                    anchor_bytes=anchor_b, slab_bytes=slab_b, batch_size=25,
                ) as session:
                    pairs = [(u, v) for b in session.iter_batches() for u, v, _, _ in b]
            expected = [(i, j) for i in range(16) for j in range(i + 1, 16)]
            self.assertEqual(sorted(pairs), expected)
            self.assertFalse(store.window_active)

    def test_scenario1_all_vs_all_windowed_exact_pairs_ram_hits_and_resume(self):
        lengths = [80 + (i % 5) * 10 for i in range(30)]
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=64, dtype=np.float16)
            total = sum(L * 64 * 2 for L in lengths)
            budget = int(total * 0.35)
            store = self._store_with_budget(path, headers, budget)
            anchor_b, slab_b = engine.plan_windowed_budgets(store)
            comp = np.zeros((30, 30), dtype=bool)

            seen = []
            with engine.WindowedPairSession.from_masks(
                store, headers, comp, None,
                anchor_bytes=anchor_b, slab_bytes=slab_b, batch_size=60,
            ) as session:
                self.assertTrue(store.window_active)
                self.assertLessEqual(store.window_cached_bytes, budget + 16384)
                for batch in session.iter_batches():
                    idxs = {u for u, _, _, _ in batch} | {v for _, v, _, _ in batch}
                    # Every batch emitted must have 100% RAM hits in anchor + active slab.
                    self.assertTrue(store.all_cached(idxs))
                    seen.extend((u, v) for u, v, _, _ in batch)
            self.assertFalse(store.window_active)
            expected = [(i, j) for i in range(30) for j in range(i + 1, 30)]
            self.assertEqual(len(seen), len(expected))
            self.assertEqual(sorted(seen), expected)

            # Mid-run resume (pretend first 150 pairs came from older row-major run)
            comp_res = np.zeros((30, 30), dtype=bool)
            for u, v in expected[:150]:
                comp_res[u, v] = True
                comp_res[v, u] = True
            store2 = self._store_with_budget(path, headers, budget)
            resumed = []
            with engine.WindowedPairSession.from_masks(
                store2, headers, comp_res, None,
                anchor_bytes=anchor_b, slab_bytes=slab_b, batch_size=60,
            ) as session:
                for batch in session.iter_batches():
                    resumed.extend((u, v) for u, v, _, _ in batch)
            self.assertEqual(sorted(resumed), expected[150:])
            self.assertFalse(store2.window_active)

    def test_scenario2_sparse_network_skips_empty_slabs_and_inactive_seqs(self):
        n = 36
        lengths = [100] * n
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=64, dtype=np.float16)
            total = n * 100 * 64 * 2
            budget = int(total * 0.30)
            store = self._store_with_budget(path, headers, budget)
            anchor_b, slab_b = engine.plan_windowed_budgets(store)

            # Only connect pairs within [0..11] and [24..35]; sequences 12..23 are completely isolated!
            req = np.zeros((n, n), dtype=bool)
            for a, b in ((0, 12), (24, 36)):
                for i in range(a, b):
                    for j in range(i + 1, b):
                        if (i + j) % 3 == 0:
                            req[i, j] = req[j, i] = True
            comp = np.zeros((n, n), dtype=bool)
            expected = [
                (i, j) for i in range(n) for j in range(i + 1, n) if req[i, j]
            ]

            seen = []
            touched_in_ram = set()
            with engine.WindowedPairSession.from_masks(
                store, headers, comp, req,
                anchor_bytes=anchor_b, slab_bytes=slab_b, batch_size=50,
            ) as session:
                for batch in session.iter_batches():
                    idxs = {u for u, _, _, _ in batch} | {v for _, v, _, _ in batch}
                    self.assertTrue(store.all_cached(idxs))
                    touched_in_ram.update(store._anchor_map.keys())
                    touched_in_ram.update(store._active_slab_map.keys())
                    seen.extend((u, v) for u, v, _, _ in batch)
                self.assertGreater(session.slabs_skipped, 0)
            self.assertEqual(sorted(seen), expected)
            # Isolated middle third (12..23) must NEVER be loaded into anchor or slab!
            self.assertTrue(touched_in_ram.isdisjoint(range(12, 24)))

    def test_scenario3_complete_and_sparse_injection_pins_hubs_and_reads_once(self):
        n_old, n_new = 24, 6
        n = n_old + n_new
        lengths = [100] * n
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=64, dtype=np.float16)
            # Each seq = 12,800 B; 6 new seqs = 76,800 B; budget = 160,000 B (anchor ~96k -> fits all 6 new!)
            store = self._store_with_budget(path, headers, 160_000)
            anchor_b, slab_b = engine.plan_windowed_budgets(store)

            # 3A: Complete bipartite + new clique
            keys_3a = np.sort(np.array(
                [u * n + v for u in range(n_old) for v in range(n_old, n)]
                + [u * n + v for u in range(n_old, n) for v in range(u + 1, n)],
                dtype=np.int64,
            ))
            seen_3a = []
            with engine.WindowedPairSession.from_pending_keys(
                store, headers, keys_3a,
                anchor_bytes=anchor_b, slab_bytes=slab_b, batch_size=40,
            ) as session:
                for batch in session.iter_batches():
                    idxs = {u for u, _, _, _ in batch} | {v for _, v, _, _ in batch}
                    self.assertTrue(store.all_cached(idxs))
                    seen_3a.extend(u * n + v for u, v, _, _ in batch)
                self.assertEqual(session.anchors_loaded, 1)
            self.assertEqual(sorted(seen_3a), keys_3a.tolist())
            # Every sequence read AT MOST ONCE (1.0x file total)!
            self.assertEqual(store.disk_reads, n)
            self.assertEqual(store.disk_read_bytes, store.total_source_bytes)

            # 3B: Sparse injection (old-old + old-new + new-new)
            rng = np.random.default_rng(101)
            sp = set()
            for u in range(n_old, n):
                for v in rng.choice(n, size=10, replace=False):
                    v = int(v)
                    if u != v:
                        a, b = (u, v) if u < v else (v, u)
                        sp.add(a * n + b)
            for _ in range(40):
                u, v = int(rng.integers(0, n_old)), int(rng.integers(0, n_old))
                if u != v:
                    a, b = (u, v) if u < v else (v, u)
                    sp.add(a * n + b)
            keys_3b = np.sort(np.array(sorted(sp), dtype=np.int64))
            store_sp = self._store_with_budget(path, headers, 160_000)
            seen_3b = []
            with engine.WindowedPairSession.from_pending_keys(
                store_sp, headers, keys_3b,
                anchor_bytes=anchor_b, slab_bytes=slab_b, batch_size=30,
            ) as session:
                for batch in session.iter_batches():
                    idxs = {u for u, _, _, _ in batch} | {v for _, v, _, _ in batch}
                    self.assertTrue(store_sp.all_cached(idxs))
                    seen_3b.extend(u * n + v for u, v, _, _ in batch)
            self.assertEqual(sorted(seen_3b), keys_3b.tolist())
            self.assertFalse(store_sp.window_active)

    def test_release_window_on_cpu_failover_frees_ram_and_finishes(self):
        lengths = [100] * 20
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=64, dtype=np.float16)
            store = self._store_with_budget(path, headers, 96_000)
            anchor_b, slab_b = engine.plan_windowed_budgets(store)
            comp = np.zeros((20, 20), dtype=bool)

            seen = []
            with engine.WindowedPairSession.from_masks(
                store, headers, comp, None,
                anchor_bytes=anchor_b, slab_bytes=slab_b, batch_size=25,
            ) as session:
                for idx_b, batch in enumerate(session.iter_batches()):
                    if idx_b == 1:
                        session.release_window()
                        self.assertFalse(store.window_active)
                        self.assertEqual(store.window_cached_bytes, 0)
                    seen.extend((u, v) for u, v, _, _ in batch)
            expected = [(i, j) for i in range(20) for j in range(i + 1, 20)]
            self.assertEqual(sorted(seen), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_gpu_fused_parity_windowed_vs_full_ram_all_3_scenarios(self):
        n = 32
        lengths = [90 + (i % 7) * 12 for i in range(n)]
        dev = torch.device("cuda:0")
        align_tool.GLOBAL_GAP_P, align_tool.LOCAL_GAP_P = 0.0, -2.0
        inject_tool.GLOBAL_GAP_P, inject_tool.LOCAL_GAP_P = 0.0, -2.0

        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=128, dtype=np.float16)
            total = sum(L * 128 * 2 for L in lengths)
            budget = int(total * 0.35)

            full_store = self._store_with_budget(path, headers, total * 2)
            self.assertTrue(full_store.fully_cached)

            def _run_gpu_batches(store, batches):
                out = []
                for b in batches:
                    out.extend(
                        engine.run_tiled_accelerator_pipeline(
                            b,
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
                    )
                return out

            # Scenario 1: Full all-vs-all
            comp = np.zeros((n, n), dtype=bool)
            all_tasks = list(align_tool._iter_pending_pairs(headers, comp, None))
            ref1 = _run_gpu_batches(full_store, [all_tasks])

            w_store1 = self._store_with_budget(path, headers, budget)
            a_b, s_b = engine.plan_windowed_budgets(w_store1, device=dev)
            with engine.WindowedPairSession.from_masks(
                w_store1, headers, comp, None, anchor_bytes=a_b, slab_bytes=s_b, batch_size=80
            ) as sess:
                got1 = _run_gpu_batches(w_store1, sess.iter_batches())
            ok1, msg1 = engine.compare_precision_results(ref1, got1, per_residue_tolerance=1e-4)
            self.assertTrue(ok1, msg1)

            # Scenario 2: Sparse
            rng = np.random.default_rng(55)
            req = np.triu(rng.random((n, n)) < 0.25, k=1)
            req = req | req.T
            sp_tasks = list(align_tool._iter_pending_pairs(headers, comp, req))
            ref2 = _run_gpu_batches(full_store, [sp_tasks])
            w_store2 = self._store_with_budget(path, headers, budget)
            with engine.WindowedPairSession.from_masks(
                w_store2, headers, comp, req, anchor_bytes=a_b, slab_bytes=s_b, batch_size=50
            ) as sess:
                got2 = _run_gpu_batches(w_store2, sess.iter_batches())
            ok2, msg2 = engine.compare_precision_results(ref2, got2, per_residue_tolerance=1e-4)
            self.assertTrue(ok2, msg2)

            # Scenario 3: Network Injection (24 old + 8 new)
            keys3 = np.sort(np.array(
                [u * n + v for u in range(24) for v in range(24, n)]
                + [u * n + v for u in range(24, n) for v in range(u + 1, n)],
                dtype=np.int64,
            ))
            inj_tasks = list(inject_tool._iter_pending_keys(headers, keys3, n))
            ref3 = _run_gpu_batches(full_store, [inj_tasks])
            w_store3 = self._store_with_budget(path, headers, budget)
            with engine.WindowedPairSession.from_pending_keys(
                w_store3, headers, keys3, anchor_bytes=a_b, slab_bytes=s_b, batch_size=50
            ) as sess:
                got3 = _run_gpu_batches(w_store3, sess.iter_batches())
            ok3, msg3 = engine.compare_precision_results(ref3, got3, per_residue_tolerance=1e-4)
            self.assertTrue(ok3, msg3)

    def _make_complete_h5(self, path, headers, arrays, model_name="esm2_t6_8m", saving_mode="float16"):
        dtype = np.float16 if saving_mode == "float16" else np.float32
        with h5py.File(path, "w") as hf:
            grp = Embedding_HDF5.create_metadata_first_file(
                hf,
                headers,
                ["A" * len(a) for a in arrays],
                model_name,
                saving_mode,
            )
            for h, a in zip(headers, arrays):
                grp.create_dataset(h, data=a.astype(dtype))
            Embedding_HDF5.mark_generation_complete(hf)

    @staticmethod
    def _read_network_tuples(net_path):
        with h5py.File(net_path, "r") as hf:
            return [
                (int(i), int(j), float(ls), int(ll), float(gs), int(gl))
                for i, j, ls, ll, gs, gl in zip(
                    hf["i"][:], hf["j"][:],
                    hf["l_score"][:], hf["l_len"][:],
                    hf["g_score"][:], hf["g_len"][:],
                )
            ]

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_tool_align_similarity_matrix_end_to_end_windowed_vs_full_ram(self):
        """End-to-end run_job_distributor() check for Scenario 1 (all-vs-all) and Scenario 2 (sparse)."""
        rng = np.random.default_rng(777)
        n = 24
        headers = [f"seq{i:02d}" for i in range(n)]
        arrays = [rng.standard_normal((80 + (i % 4) * 10, 128)).astype(np.float16) for i in range(n)]
        total_bytes = sum(a.nbytes for a in arrays)
        cuda_cand = Hardware_Utils.DeviceCandidate("cuda:0", "CUDA", torch.device("cuda:0"), "cuda")
        tiled_plan = Hardware_Utils.BenchmarkResult(cuda_cand, 1000.0, lanes=1, variant="tiled")

        for label, prefilter, strength in (("all_vs_all", False, 0), ("sparse", True, 65)):
            with self.subTest(mode=label), tempfile.TemporaryDirectory() as tmp:
                embed_dir = os.path.join(tmp, "embed")
                os.makedirs(embed_dir)
                h5_name = f"toy_{label}_[esm2_t6_8m]_embeddings.h5"
                self._make_complete_h5(os.path.join(embed_dir, h5_name), headers, arrays)

                outputs = {}
                for mode, budget in (("full", total_bytes * 4), ("windowed", int(total_bytes * 0.35))):
                    net_dir = os.path.join(tmp, f"net_{mode}")
                    os.makedirs(net_dir)
                    out_buf = io.StringIO()
                    saved_globals = (
                        "MODEL_NAME", "SEQUENCE_SET", "FULL_INPUT_HDF5",
                        "RESULTS_DIR", "FINAL_OUTPUT_NET",
                        "active_matmul_precision", "active_embedding_store",
                        "active_sequence_lengths",
                    )
                    with mock.patch.multiple(
                        align_tool,
                        **{k: getattr(align_tool, k) for k in saved_globals},
                        INPUT_HDF5=h5_name,
                        EMBED_DIR=embed_dir,
                        NETWORK_DIR=net_dir,
                        EDGE_PREFILTERING=prefilter,
                        PREFILTER_STRENGTH=strength,
                        BATCH_SIZE=60,
                        WORKERS=2,
                        HOST_CACHE_GB="auto",
                        DEVICE_SELECTION="cuda:0",
                        EXECUTION_MODE="tiled",
                        ACCELERATOR_PRECISION="ieee_fp32",
                        LOCAL_GAP_P=-2.0,
                        GLOBAL_GAP_P=0.0,
                        set_start_method=mock.DEFAULT,
                        _benchmark_processing_plans=mock.Mock(return_value=[tiled_plan]),
                    ), mock.patch.object(
                        engine, "resolve_host_cache_bytes", return_value=budget
                    ), redirect_stdout(out_buf), redirect_stderr(io.StringIO()):
                        rc = align_tool.run_job_distributor()
                        store_after = align_tool.active_embedding_store
                    self.assertIsNone(rc)
                    self.assertFalse(store_after.window_active)
                    printed = out_buf.getvalue()
                    if mode == "windowed":
                        self.assertIn("[Memory] Activating partial-RAM sliding window", printed)
                    else:
                        self.assertNotIn("Activating partial-RAM sliding window", printed)
                    net_path = os.path.join(net_dir, f"toy_{label}_[esm2_t6_8m]_network.h5")
                    outputs[mode] = self._read_network_tuples(net_path)

                self.assertGreater(len(outputs["full"]), 0)
                self.assertEqual(
                    sorted((r[0], r[1]) for r in outputs["full"]),
                    sorted((r[0], r[1]) for r in outputs["windowed"]),
                )
                ok, msg = engine.compare_precision_results(
                    outputs["full"], outputs["windowed"], per_residue_tolerance=1e-4
                )
                self.assertTrue(ok, msg)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_tool_network_injection_end_to_end_windowed_vs_full_ram(self):
        """End-to-end run_injection() check for Scenario 3A (complete) and Scenario 3B (sparse)."""
        rng = np.random.default_rng(888)
        n_old, n_new = 18, 6
        n = n_old + n_new
        old_headers = [f"s{i:02d}" for i in range(n_old)]
        new_headers = [f"s{i:02d}" for i in range(n)]
        arrays = [rng.standard_normal((80 + (i % 4) * 10, 128)).astype(np.float16) for i in range(n)]
        total_bytes = sum(a.nbytes for a in arrays)
        cuda_cand = Hardware_Utils.DeviceCandidate("cuda:0", "CUDA", torch.device("cuda:0"), "cuda")
        tiled_plan = Hardware_Utils.BenchmarkResult(cuda_cand, 1000.0, lanes=1, variant="tiled")

        for kind, sparse_old in (("complete", False), ("sparse", True)):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                embed_dir = os.path.join(tmp, "embed")
                os.makedirs(embed_dir)
                emb_name = f"new_{kind}_[esm2_t6_8m]_embeddings.h5"
                self._make_complete_h5(os.path.join(embed_dir, emb_name), new_headers, arrays)

                if sparse_old:
                    edges = [
                        (i, j) for i in range(n_old) for j in range(i + 1, n_old)
                        if (i + j) % 3 == 0
                    ]
                else:
                    edges = [(i, j) for i in range(n_old) for j in range(i + 1, n_old)]

                outputs = {}
                for mode, budget in (("full", total_bytes * 4), ("windowed", int(total_bytes * 0.45))):
                    net_dir = os.path.join(tmp, f"net_{mode}")
                    os.makedirs(net_dir)
                    old_net_name = f"old_{kind}_[esm2_t6_8m]_network.h5"
                    with h5py.File(os.path.join(net_dir, old_net_name), "w") as hf:
                        hf.attrs["model_name"] = "esm2_t6_8m"
                        hf.attrs["saving_mode"] = "float16"
                        hf.attrs["gap_penalties"] = np.asarray([-2.0, 0.0], np.float32)
                        hf.attrs["matmul_precision"] = "ieee_fp32"
                        hf.create_dataset(
                            "headers",
                            data=np.array(old_headers, dtype=object),
                            dtype=h5py.string_dtype(encoding="utf-8"),
                        )
                        hf.create_dataset(
                            "seq_lens",
                            data=np.asarray([len(arrays[i]) for i in range(n_old)], dtype=np.uint16),
                        )
                        cnt = len(edges)
                        hf.create_dataset("i", data=np.asarray([e[0] for e in edges], dtype=np.uint16))
                        hf.create_dataset("j", data=np.asarray([e[1] for e in edges], dtype=np.uint16))
                        hf.create_dataset("l_score", data=np.arange(cnt, dtype=np.float32) + 5.0)
                        hf.create_dataset("l_len", data=np.full(cnt, 20, dtype=np.uint16))
                        hf.create_dataset("g_score", data=np.arange(cnt, dtype=np.float32) + 10.0)
                        hf.create_dataset("g_len", data=np.full(cnt, 40, dtype=np.uint16))
                        if sparse_old:
                            Prefilter_Calibration.write_record(
                                hf,
                                Prefilter_Calibration.PrefilterRecord(
                                    strength=65.0,
                                    threshold=0.5,
                                    pooling="max",
                                    length_power=2.0,
                                    kept=cnt,
                                    skipped=(n_old * (n_old - 1)) // 2 - cnt,
                                    sample=Prefilter_Calibration.CalibrationSample(
                                        np.asarray([e[0] for e in edges[:5]], dtype=np.uint16),
                                        np.asarray([e[1] for e in edges[:5]], dtype=np.uint16),
                                        np.ones(min(5, cnt), dtype=np.float32),
                                    ),
                                ),
                                np.uint16,
                            )

                    out_buf = io.StringIO()
                    saved_globals = (
                        "_old_seq_set", "_new_seq_set", "_model_name",
                        "RESULTS_DIR", "FINAL_OUTPUT_NET", "CONFIG_FILE",
                        "LOCAL_GAP_P", "GLOBAL_GAP_P",
                    )
                    with mock.patch.multiple(
                        inject_tool,
                        **{k: getattr(inject_tool, k) for k in saved_globals},
                        OLD_NETWORK=old_net_name,
                        NEW_EMBEDDINGS=emb_name,
                        EMBED_DIR=embed_dir,
                        NETWORK_DIR=net_dir,
                        BATCH_SIZE=50,
                        WORKERS=2,
                        HOST_CACHE_GB="auto",
                        DEVICE_SELECTION="cuda:0",
                        EXECUTION_MODE="tiled",
                        set_start_method=mock.DEFAULT,
                        _benchmark_injection_plans=mock.Mock(return_value=[tiled_plan]),
                    ), mock.patch.object(
                        engine, "resolve_host_cache_bytes", return_value=budget
                    ), redirect_stdout(out_buf), redirect_stderr(io.StringIO()):
                        inject_tool.run_injection()
                    printed = out_buf.getvalue()
                    if mode == "windowed":
                        self.assertIn("[Memory] Activating partial-RAM sliding window", printed)
                    else:
                        self.assertNotIn("Activating partial-RAM sliding window", printed)
                    net_path = os.path.join(net_dir, f"new_{kind}_[esm2_t6_8m]_network.h5")
                    outputs[mode] = self._read_network_tuples(net_path)

                self.assertGreater(len(outputs["full"]), 0)
                self.assertEqual(
                    sorted((r[0], r[1]) for r in outputs["full"]),
                    sorted((r[0], r[1]) for r in outputs["windowed"]),
                )
                ok, msg = engine.compare_precision_results(
                    outputs["full"], outputs["windowed"], per_residue_tolerance=1e-4
                )
                self.assertTrue(ok, msg)

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

    def test_streamed_pair_similarity_exact_percentile_keys_and_edge_cases(self):
        import tracemalloc

        rng = np.random.default_rng(2026)
        n = 140
        pooled = rng.standard_normal((n, 32)).astype(np.float32)
        pooled[7] = 0.0  # zero vector
        lengths = rng.integers(30, 300, n)
        lengths[11] = 0  # zero length

        # Test normal bracket, tiny bracket forcing refinement, and multiple block sizes.
        for block, max_br in ((4_000_000, 4_000_000), (140 * 17, 25), (140 * 9, 5)):
            with self.subTest(block=block, max_br=max_br), mock.patch.multiple(
                Prefilter_Calibration,
                _SIMILARITY_BLOCK=block,
                _MAX_BRACKET_VALUES=max_br,
            ):
                sim = Prefilter_Calibration.StreamedPairSimilarity(pooled, lengths, 2.0)
                # Materialize exact streamed upper triangle to verify 100% bit-exact percentile & keys.
                upper_vals = []
                upper_keys = []
                for s, e, blk in sim.iter_upper_blocks():
                    m = sim._upper_mask(s, e, blk.shape[1])
                    rr, cc = np.nonzero(m)
                    upper_vals.append(blk[rr, cc])
                    upper_keys.append((rr.astype(np.int64) + s) * n + (cc.astype(np.int64) + s))
                all_vals = np.concatenate(upper_vals)
                all_keys = np.concatenate(upper_keys)

                for q in (0.0, 1.5, 25.0, 50.0, 75.0, 95.0, 99.0, 100.0):
                    want_t = float(np.percentile(all_vals, q))
                    got_t = sim.upper_percentile(q)
                    self.assertEqual(got_t, want_t, f"q={q}")
                    got_t2, got_keys = sim.upper_percentile_and_keys(q)
                    self.assertEqual(got_t2, want_t, f"q={q}")
                    want_keys = all_keys[all_vals >= want_t]
                    np.testing.assert_array_equal(got_keys, want_keys)
                    self.assertEqual(sim.count_upper_at_or_above(want_t), int(want_keys.size))

                # Also check proximity to dense full-matrix length_adjusted_similarity (<= 1e-6 float32 ULP).
                dense = Prefilter_Calibration.length_adjusted_similarity(pooled, lengths, 2.0)
                rr_sample = rng.integers(0, n, size=200)
                cc_sample = rng.integers(0, n, size=200)
                np.testing.assert_allclose(
                    sim.at(rr_sample, cc_sample), dense[rr_sample, cc_sample], atol=1e-6
                )

        # All-identical vectors (100% ties) and tiny n < 2 edge cases.
        tied_pooled = np.ones((80, 16), dtype=np.float32)
        tied_lens = np.full(80, 100, dtype=np.int32)
        with mock.patch.object(Prefilter_Calibration, "_MAX_BRACKET_VALUES", 10):
            tied_sim = Prefilter_Calibration.StreamedPairSimilarity(tied_pooled, tied_lens, 2.0)
            t_tied, k_tied = tied_sim.upper_percentile_and_keys(50.0)
            self.assertAlmostEqual(t_tied, 1.0, places=6)
            self.assertEqual(k_tied.size, 80 * 79 // 2)

        single = Prefilter_Calibration.StreamedPairSimilarity(pooled[:1], lengths[:1], 2.0)
        self.assertIsNone(single.upper_percentile(50.0))
        self.assertEqual(single.upper_percentile_and_keys(50.0)[1].size, 0)

        # Memory check: N=1500 dense would allocate ~27 MB (12*n^2); streamed stays well below dense.
        n_mem = 1500
        p_mem = rng.standard_normal((n_mem, 16)).astype(np.float32)
        l_mem = rng.integers(50, 200, n_mem)
        with mock.patch.object(Prefilter_Calibration, "_SIMILARITY_BLOCK", 60_000):
            tracemalloc.start()
            try:
                s_mem = Prefilter_Calibration.StreamedPairSimilarity(p_mem, l_mem, 2.0)
                _t_m, _k_m = s_mem.upper_percentile_and_keys(99.0)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        # Dense 12*n^2 would be ~27 MB; streamed peak should stay under 6 MB.
        self.assertLess(peak, 6 * 1024 * 1024)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_sparse_align_uses_int64_keys_with_resume_and_dedup(self):
        """Sparse run_job_distributor() resumes & deduplicates purely with int64 keys (no N x N masks)."""
        rng = np.random.default_rng(909)
        n = 28
        headers = [f"p{i:02d}" for i in range(n)]
        arrays = [rng.standard_normal((80 + (i % 5) * 10, 128)).astype(np.float16) for i in range(n)]
        total_bytes = sum(a.nbytes for a in arrays)
        cuda_cand = Hardware_Utils.DeviceCandidate("cuda:0", "CUDA", torch.device("cuda:0"), "cuda")
        tiled_plan = Hardware_Utils.BenchmarkResult(cuda_cand, 1000.0, lanes=1, variant="tiled")

        with tempfile.TemporaryDirectory() as tmp:
            embed_dir = os.path.join(tmp, "embed")
            os.makedirs(embed_dir)
            emb_name = "sparse_keys_[esm2_t6_8m]_embeddings.h5"
            emb_path = os.path.join(embed_dir, emb_name)
            self._make_complete_h5(emb_path, headers, arrays)
            checksum = align_tool.calculate_file_hash(emb_path)

            # First get reference required keys from prefilter_pair_keys.
            with mock.patch.multiple(
                align_tool, FULL_INPUT_HDF5=emb_path, PREFILTER_STRENGTH=75
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                req_keys, cut = align_tool.prefilter_pair_keys(headers, [len(a) for a in arrays])
            self.assertGreater(req_keys.size, 6)

            # Seed batch_00000.h5 with first 4 required pairs (plus 1 duplicate and 1 non-required pair!).
            net_dir = os.path.join(tmp, "net")
            temp_dir = os.path.join(net_dir, "sparse_keys_[esm2_t6_8m]_network_temp")
            os.makedirs(temp_dir)
            seeded_req = req_keys[:4]
            non_req = next(
                k for k in range(n * n)
                if (k // n) < (k % n) and not np.isin(k, req_keys)
            )
            seed_keys = np.array(
                [seeded_req[0], seeded_req[1], seeded_req[0], non_req, seeded_req[2], seeded_req[3]],
                dtype=np.int64,
            )
            with h5py.File(os.path.join(temp_dir, "batch_00000.h5"), "w") as hf:
                hf.attrs["embedding_checksum"] = checksum
                hf.attrs["model_name"] = "esm2_t6_8m"
                hf.attrs["gap_penalties"] = np.asarray([-2.0, 0.0], dtype=np.float32)
                hf.attrs["matmul_precision"] = "ieee_fp32"
                hf.create_dataset("i", data=(seed_keys // n).astype(np.uint16))
                hf.create_dataset("j", data=(seed_keys % n).astype(np.uint16))
                hf.create_dataset("l_score", data=np.full(seed_keys.size, 123.5, dtype=np.float32))
                hf.create_dataset("l_len", data=np.full(seed_keys.size, 15, dtype=np.uint16))
                hf.create_dataset("g_score", data=np.full(seed_keys.size, 456.25, dtype=np.float32))
                hf.create_dataset("g_len", data=np.full(seed_keys.size, 30, dtype=np.uint16))

            out_buf = io.StringIO()
            saved_globals = (
                "MODEL_NAME", "SEQUENCE_SET", "FULL_INPUT_HDF5",
                "RESULTS_DIR", "FINAL_OUTPUT_NET",
                "active_matmul_precision", "active_embedding_store",
                "active_sequence_lengths",
            )
            with mock.patch.multiple(
                align_tool,
                **{k: getattr(align_tool, k) for k in saved_globals},
                INPUT_HDF5=emb_name,
                EMBED_DIR=embed_dir,
                NETWORK_DIR=net_dir,
                EDGE_PREFILTERING=True,
                PREFILTER_STRENGTH=75,
                BATCH_SIZE=40,
                WORKERS=2,
                HOST_CACHE_GB="auto",
                DEVICE_SELECTION="cuda:0",
                EXECUTION_MODE="tiled",
                ACCELERATOR_PRECISION="ieee_fp32",
                LOCAL_GAP_P=-2.0,
                GLOBAL_GAP_P=0.0,
                set_start_method=mock.DEFAULT,
                _benchmark_processing_plans=mock.Mock(return_value=[tiled_plan]),
            ), mock.patch.object(
                engine, "resolve_host_cache_bytes", return_value=int(total_bytes * 0.35)
            ), redirect_stdout(out_buf), redirect_stderr(io.StringIO()):
                rc = align_tool.run_job_distributor()

            self.assertIsNone(rc)
            printed = out_buf.getvalue()
            self.assertIn(f"Already computed pairs: 4\n", printed)
            self.assertIn(f"Pairs queued for calculation: {req_keys.size - 4}\n", printed)
            self.assertIn("[Memory] Activating partial-RAM sliding window", printed)
            net_path = os.path.join(net_dir, "sparse_keys_[esm2_t6_8m]_network.h5")
            rows = self._read_network_tuples(net_path)
            got_keys = sorted(r[0] * n + r[1] for r in rows)
            self.assertEqual(got_keys, req_keys.tolist())
            # Seeded pairs retained exact seeded scores and duplicate/non-required dropped.
            by_key = {r[0] * n + r[1]: r for r in rows}
            self.assertNotIn(non_req, by_key)
            for sk in seeded_req.tolist():
                self.assertAlmostEqual(by_key[sk][2], 123.5, places=4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_network_injection_streamed_similarity_parity_with_dense(self):
        """Forcing StreamedPairSimilarity in Network_Injection matches dense pair_similarity."""
        rng = np.random.default_rng(1010)
        n_old, n_new = 20, 6
        n = n_old + n_new
        old_headers = [f"q{i:02d}" for i in range(n_old)]
        new_headers = [f"q{i:02d}" for i in range(n)]
        arrays = [rng.standard_normal((80 + (i % 4) * 10, 128)).astype(np.float16) for i in range(n)]
        total_bytes = sum(a.nbytes for a in arrays)
        cuda_cand = Hardware_Utils.DeviceCandidate("cuda:0", "CUDA", torch.device("cuda:0"), "cuda")
        tiled_plan = Hardware_Utils.BenchmarkResult(cuda_cand, 1000.0, lanes=1, variant="tiled")
        edges = [(i, j) for i in range(n_old) for j in range(i + 1, n_old) if (i + j) % 3 == 0]

        with tempfile.TemporaryDirectory() as tmp:
            embed_dir = os.path.join(tmp, "embed")
            os.makedirs(embed_dir)
            emb_name = "inj_stream_[esm2_t6_8m]_embeddings.h5"
            self._make_complete_h5(os.path.join(embed_dir, emb_name), new_headers, arrays)

            outputs = {}
            for mode, threshold_bytes in (("dense", 1 << 40), ("streamed", 1)):
                net_dir = os.path.join(tmp, f"net_{mode}")
                os.makedirs(net_dir)
                old_name = "old_stream_[esm2_t6_8m]_network.h5"
                with h5py.File(os.path.join(net_dir, old_name), "w") as hf:
                    hf.attrs["model_name"] = "esm2_t6_8m"
                    hf.attrs["saving_mode"] = "float16"
                    hf.attrs["gap_penalties"] = np.asarray([-2.0, 0.0], dtype=np.float32)
                    hf.attrs["matmul_precision"] = "ieee_fp32"
                    hf.create_dataset(
                        "headers",
                        data=np.array(old_headers, dtype=object),
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
                    hf.create_dataset(
                        "seq_lens",
                        data=np.asarray([len(arrays[i]) for i in range(n_old)], dtype=np.uint16),
                    )
                    cnt = len(edges)
                    hf.create_dataset("i", data=np.asarray([e[0] for e in edges], dtype=np.uint16))
                    hf.create_dataset("j", data=np.asarray([e[1] for e in edges], dtype=np.uint16))
                    hf.create_dataset("l_score", data=np.arange(cnt, dtype=np.float32) + 5.0)
                    hf.create_dataset("l_len", data=np.full(cnt, 20, dtype=np.uint16))
                    hf.create_dataset("g_score", data=np.arange(cnt, dtype=np.float32) + 10.0)
                    hf.create_dataset("g_len", data=np.full(cnt, 40, dtype=np.uint16))

                saved_globals = (
                    "_old_seq_set", "_new_seq_set", "_model_name",
                    "RESULTS_DIR", "FINAL_OUTPUT_NET", "CONFIG_FILE",
                    "LOCAL_GAP_P", "GLOBAL_GAP_P",
                )
                with mock.patch.multiple(
                    inject_tool,
                    **{k: getattr(inject_tool, k) for k in saved_globals},
                    OLD_NETWORK=old_name,
                    NEW_EMBEDDINGS=emb_name,
                    EMBED_DIR=embed_dir,
                    NETWORK_DIR=net_dir,
                    BATCH_SIZE=50,
                    WORKERS=2,
                    HOST_CACHE_GB="auto",
                    DEVICE_SELECTION="cuda:0",
                    EXECUTION_MODE="tiled",
                    set_start_method=mock.DEFAULT,
                    _benchmark_injection_plans=mock.Mock(return_value=[tiled_plan]),
                ), mock.patch.object(
                    Prefilter_Calibration, "STREAMED_SIMILARITY_BYTES", threshold_bytes
                ), mock.patch.object(
                    engine, "resolve_host_cache_bytes", return_value=int(total_bytes * 0.45)
                ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    inject_tool.run_injection()
                net_path = os.path.join(net_dir, "inj_stream_[esm2_t6_8m]_network.h5")
                outputs[mode] = self._read_network_tuples(net_path)

            self.assertEqual(
                sorted((r[0], r[1]) for r in outputs["dense"]),
                sorted((r[0], r[1]) for r in outputs["streamed"]),
            )
            ok, msg = engine.compare_precision_results(
                outputs["dense"], outputs["streamed"], per_residue_tolerance=1e-4
            )
            self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
