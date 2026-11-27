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
            expected = []
            for a, b in ((0, 12), (24, 36)):
                for i in range(a, b):
                    for j in range(i + 1, b):
                        if (i + j) % 3 == 0:
                            expected.append((i, j))
            req_keys = np.array([i * n + j for i, j in expected], dtype=np.int64)

            seen = []
            touched_in_ram = set()
            with engine.WindowedPairSession.from_pending_keys(
                store, headers, req_keys,
                anchor_bytes=anchor_b, slab_bytes=slab_b, batch_size=50,
            ) as session:
                for batch in session.iter_batches():
                    idxs = {u for u, _, _, _ in batch} | {v for _, v, _, _ in batch}
                    self.assertTrue(store.all_cached(idxs))
                    touched_in_ram.update(store._anchor_map.keys())
                    touched_in_ram.update(store._active_slab_map.keys())
                    seen.extend((u, v) for u, v, _, _ in batch)
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
            all_tasks = list(align_tool._iter_pending_pairs(headers, comp))
            ref1 = _run_gpu_batches(full_store, [all_tasks])

            w_store1 = self._store_with_budget(path, headers, budget)
            a_b, s_b = engine.plan_windowed_budgets(w_store1, device=dev)
            with engine.WindowedPairSession.from_masks(
                w_store1, headers, comp, None, anchor_bytes=a_b, slab_bytes=s_b, batch_size=80
            ) as sess:
                got1 = _run_gpu_batches(w_store1, sess.iter_batches())
            ok1, msg1 = engine.compare_precision_results(ref1, got1, per_residue_tolerance=1e-4)
            self.assertTrue(ok1, msg1)

            # Scenario 2: Sparse (int64 required_keys)
            rng = np.random.default_rng(55)
            req = np.triu(rng.random((n, n)) < 0.25, k=1)
            rr, cc = np.nonzero(req)
            sp_keys = rr.astype(np.int64) * n + cc.astype(np.int64)
            sp_tasks = list(align_tool._iter_pending_key_tasks(headers, sp_keys, n))
            ref2 = _run_gpu_batches(full_store, [sp_tasks])
            w_store2 = self._store_with_budget(path, headers, budget)
            with engine.WindowedPairSession.from_pending_keys(
                w_store2, headers, sp_keys, anchor_bytes=a_b, slab_bytes=s_b, batch_size=50
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

    def test_vram_single_batch_skip_and_prefilter_pair_keys_dense_parity(self):
        # 1) Single-batch VRAM-fit skips windowing; multi-batch still activates windowing.
        lengths = [100] * 20
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=64, dtype=np.float16)
            # total source = 256,000 B; total fp32 = 512,000 B; budget = 102,400 B (40%)
            store = self._store_with_budget(path, headers, 102_400)
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

    def test_scalar_winner_releases_ram_window_immediately(self):
        # Simulate scalar CUDA plan winning benchmark: window must be released before batch work.
        rng = np.random.default_rng(1212)
        n = 16
        headers = [f"x{i:02d}" for i in range(n)]
        arrays = [rng.standard_normal((80, 64)).astype(np.float16) for _ in range(n)]
        total_bytes = sum(a.nbytes for a in arrays)
        cuda_cand = Hardware_Utils.DeviceCandidate("cuda:0", "CUDA", torch.device("cuda:0"), "cuda")
        scalar_plan = Hardware_Utils.BenchmarkResult(cuda_cand, 500.0, lanes=1, variant="scalar")
        window_active_during_batch = []

        def fake_run_batch(ranked_plans, active_plan_index, batch, *args, **kwargs):
            window_active_during_batch.append(align_tool.active_embedding_store.window_active)
            return active_plan_index

        with tempfile.TemporaryDirectory() as tmp:
            embed_dir = os.path.join(tmp, "embed")
            net_dir = os.path.join(tmp, "net")
            os.makedirs(embed_dir)
            os.makedirs(net_dir)
            emb_name = "scalar_rel_[esm2_t6_8m]_embeddings.h5"
            self._make_complete_h5(os.path.join(embed_dir, emb_name), headers, arrays)
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
                EDGE_PREFILTERING=False,
                BATCH_SIZE=30,
                WORKERS=2,
                HOST_CACHE_GB="auto",
                DEVICE_SELECTION="cuda:0",
                EXECUTION_MODE="auto",
                ACCELERATOR_PRECISION="ieee_fp32",
                LOCAL_GAP_P=-2.0,
                GLOBAL_GAP_P=0.0,
                set_start_method=mock.DEFAULT,
                _validate_execution_mode_hardware=mock.Mock(),
                _benchmark_processing_plans=mock.Mock(return_value=[scalar_plan]),
                _run_batch_with_ranked_plans=mock.Mock(side_effect=fake_run_batch),
                compile_final_output=mock.Mock(),
            ), mock.patch.object(
                align_tool.Hardware_Utils, "get_available_devices", return_value=[cuda_cand]
            ), mock.patch.object(
                engine, "resolve_host_cache_bytes", return_value=int(total_bytes * 0.35)
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                align_tool.run_job_distributor()
        self.assertTrue(window_active_during_batch)
        self.assertTrue(all(not active for active in window_active_during_batch))

    def test_prefetcher_startup_hdf5_failure_falls_back_and_never_hangs(self):
        """D-HI-1: HDF5 open failure in _WindowSlabPrefetcher never hangs submit() or wait_result()."""
        lengths = [64] * 12
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=32, dtype=np.float16)
            store = self._store_with_budget(path, headers, 24_000)

            # 1) open_window() waits for prefetcher readiness, detects startup OSError,
            #    cleans up buffers, and returns False cleanly.
            with mock.patch.object(
                engine.h5py, "File", side_effect=OSError("simulated hdf5 open failure")
            ):
                opened = store.open_window(anchor_rows_cap=128, slab_rows_cap=128)
            self.assertFalse(opened)
            self.assertFalse(store.window_active)
            self.assertEqual(store.window_cached_bytes, 0)
            self.assertIsNone(store._prefetcher)

            # 2) Direct _WindowSlabPrefetcher failure before submit() raises on submit()
            #    and never blocks in wait_result().
            slab_bufs = [
                np.empty((128, 32), dtype=np.float16),
                np.empty((128, 32), dtype=np.float16),
            ]
            with mock.patch.object(
                engine.h5py, "File", side_effect=OSError("cannot open hdf5")
            ):
                prefetcher = engine._WindowSlabPrefetcher(store, slab_bufs)
                prefetcher.start()
                self.assertTrue(prefetcher._ready.wait(timeout=5.0))
                prefetcher.join(timeout=5.0)
                with self.assertRaisesRegex(OSError, "cannot open hdf5"):
                    prefetcher.submit(0, [0, 1])
                # Even if _pending were already set when thread exited, wait_result() raises promptly.
                prefetcher._pending = True
                with self.assertRaisesRegex(OSError, "cannot open hdf5"):
                    prefetcher.wait_result()
                self.assertFalse(prefetcher._pending)

                # Dead thread with no recorded fatal error also raises RuntimeError promptly.
                prefetcher._fatal_error = None
                prefetcher._pending = True
                with self.assertRaises(RuntimeError):
                    prefetcher.wait_result()
                self.assertFalse(prefetcher._pending)
                prefetcher.close()

    def test_network_injection_streamed_s_min_edge_always_kept(self):
        """D-HI-2: Network_Injection s_min cut edge is always retained by upper_keys_at_or_above."""
        rng = np.random.default_rng(31415)
        for trial in range(5):
            n = 60
            pooled = rng.standard_normal((n, 128)).astype(np.float32)
            lengths = rng.integers(40, 350, n)
            sim = Prefilter_Calibration.StreamedPairSimilarity(pooled, lengths, 2.0)

            rr, cc = np.triu_indices(n, k=1)
            at_vals = sim.at(rr, cc)

            # Network_Injection set_by == "s_min": pick low-similarity old edges so s_min < t'.
            order = np.argsort(at_vals)
            edge_idx = order[10:25]  # includes low-similarity edge below 50th percentile
            edge_u = rr[edge_idx]
            edge_v = cc[edge_idx]
            skip = np.zeros(edge_u.size, dtype=bool)
            _cut, s_min, lowest_edge, _t_prime, set_by = inject_tool._similarity_cut(
                sim, edge_u, edge_v, skip, kept_share=0.20
            )
            self.assertEqual(set_by, "s_min")
            u_star = int(edge_u[lowest_edge])
            v_star = int(edge_v[lowest_edge])
            star_key = min(u_star, v_star) * n + max(u_star, v_star)
            keys = sim.upper_keys_at_or_above(s_min)
            self.assertIn(star_key, set(keys.tolist()), f"trial={trial} dropped s_min edge")

    def test_align_midrun_tiled_to_cpu_and_scalar_failover_releases_window_first(self):
        """Medium #1: Align_Similarity_Matrix releases RAM window BEFORE CPU/scalar failover executes."""
        lengths = [80] * 16
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=32, dtype=np.float16)
            store = self._store_with_budget(path, headers, 32_000)
            anchor_b, slab_b = engine.plan_windowed_budgets(store)
            comp = np.zeros((16, 16), dtype=bool)
            cuda_cand = Hardware_Utils.DeviceCandidate("cuda:0", "CUDA", torch.device("cuda:0"), "cuda")
            cpu_cand = Hardware_Utils.DeviceCandidate("cpu", "CPU", torch.device("cpu"), "cpu")
            ranked = [
                Hardware_Utils.BenchmarkResult(cuda_cand, 500.0, lanes=1, variant="tiled"),
                Hardware_Utils.BenchmarkResult(cpu_cand, 50.0, lanes=1, variant="scalar"),
            ]

            window_states = []

            def fake_process_batch(*args, **kwargs):
                window_states.append((kwargs["device"].type, store.window_active))
                if kwargs["device"].type == "cuda":
                    raise RuntimeError("simulated mid-run tiled failure")

            with engine.WindowedPairSession.from_masks(
                store, headers, comp, None, anchor_bytes=anchor_b, slab_bytes=slab_b, batch_size=30
            ) as window_session:
                self.assertTrue(store.window_active)
                with mock.patch.object(align_tool, "DEVICE_SELECTION", "auto"), \
                     mock.patch.object(align_tool, "process_batch", side_effect=fake_process_batch), \
                     redirect_stdout(io.StringIO()):
                    next_idx = align_tool._run_batch_with_ranked_plans(
                        ranked, 0, [(0, 1, "s000", "s001")], 0, 2, path, "chk",
                        window_session=window_session,
                    )
                self.assertEqual(next_idx, 1)
                self.assertEqual(window_states, [("cuda", True), ("cpu", False)])
                self.assertFalse(store.window_active)

            # Also verify internal tiled->scalar OOM fallback inside process_accelerated_tasks.
            scalar_window_state = []

            def fake_scalar_pipeline(*args, **kwargs):
                scalar_window_state.append(store.window_active)
                return [(0, 1, 1.0, 1, 2.0, 1)]

            fake_backend = mock.Mock(device_type="cuda")
            fake_backend.is_out_of_memory.return_value = True
            with engine.WindowedPairSession.from_masks(
                store, headers, comp, None, anchor_bytes=anchor_b, slab_bytes=slab_b, batch_size=30
            ) as window_session:
                self.assertTrue(store.window_active)
                with mock.patch.object(align_tool, "EXECUTION_MODE", "auto"), \
                     mock.patch.object(align_tool, "tiled_accelerator_support", return_value=(True, "ok")), \
                     mock.patch.object(align_tool, "get_accelerator_backend", return_value=fake_backend), \
                     mock.patch.object(align_tool, "cuda_memory_plan", return_value=mock.Mock(matrix_bytes=64 * 1024 * 1024)), \
                     mock.patch.object(align_tool, "run_tiled_accelerator_pipeline", side_effect=RuntimeError("CUDA out of memory")), \
                     mock.patch.object(align_tool, "_run_accelerated_pipeline", side_effect=fake_scalar_pipeline), \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    res = align_tool.process_accelerated_tasks(
                        [(0, 1, "s000", "s001")],
                        2,
                        path,
                        torch.device("cuda:0"),
                        0,
                        accelerator_workers=1,
                        embedding_store=store,
                        sequence_lengths=lengths,
                        execution_variant="tiled",
                        window_session=window_session,
                    )
                self.assertEqual(len(res), 1)
                self.assertEqual(scalar_window_state, [False])
                self.assertFalse(store.window_active)
                self.assertFalse(window_session._window_enabled)

    def test_network_injection_skips_window_for_non_tiled_accelerators(self):
        """Medium #4: Network_Injection must not open a partial-RAM window when only MPS/non-tiled exists."""
        rng = np.random.default_rng(4242)
        n_old, n_new = 12, 4
        n = n_old + n_new
        old_headers = [f"m{i:02d}" for i in range(n_old)]
        new_headers = [f"m{i:02d}" for i in range(n)]
        arrays = [rng.standard_normal((80, 64)).astype(np.float16) for _ in range(n)]
        total_bytes = sum(a.nbytes for a in arrays)
        mps_cand = Hardware_Utils.DeviceCandidate("mps", "Apple MPS", torch.device("cpu"), "mps")
        scalar_plan = Hardware_Utils.BenchmarkResult(mps_cand, 200.0, lanes=1, variant="scalar")

        with tempfile.TemporaryDirectory() as tmp:
            embed_dir = os.path.join(tmp, "embed")
            net_dir = os.path.join(tmp, "net")
            os.makedirs(embed_dir)
            os.makedirs(net_dir)
            emb_name = "mps_skip_[esm2_t6_8m]_embeddings.h5"
            self._make_complete_h5(os.path.join(embed_dir, emb_name), new_headers, arrays)
            old_name = "old_mps_[esm2_t6_8m]_network.h5"
            edges = [(i, j) for i in range(n_old) for j in range(i + 1, n_old)]
            with h5py.File(os.path.join(net_dir, old_name), "w") as hf:
                hf.attrs["model_name"] = "esm2_t6_8m"
                hf.attrs["saving_mode"] = "float16"
                hf.attrs["gap_penalties"] = np.asarray([-2.0, 0.0], dtype=np.float32)
                hf.attrs["matmul_precision"] = "ieee_fp32"
                hf.create_dataset("headers", data=np.array(old_headers, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
                hf.create_dataset("seq_lens", data=np.full(n_old, 80, dtype=np.uint16))
                cnt = len(edges)
                hf.create_dataset("i", data=np.asarray([e[0] for e in edges], dtype=np.uint16))
                hf.create_dataset("j", data=np.asarray([e[1] for e in edges], dtype=np.uint16))
                hf.create_dataset("l_score", data=np.ones(cnt, dtype=np.float32))
                hf.create_dataset("l_len", data=np.full(cnt, 20, dtype=np.uint16))
                hf.create_dataset("g_score", data=np.ones(cnt, dtype=np.float32))
                hf.create_dataset("g_len", data=np.full(cnt, 40, dtype=np.uint16))

            out_buf = io.StringIO()
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
                BATCH_SIZE=20,
                WORKERS=2,
                HOST_CACHE_GB="auto",
                DEVICE_SELECTION="auto",
                EXECUTION_MODE="auto",
                set_start_method=mock.DEFAULT,
                _benchmark_injection_plans=mock.Mock(return_value=[scalar_plan]),
                process_batch=mock.Mock(),
                compile_final_output=mock.Mock(),
                plan_windowed_budgets=mock.Mock(side_effect=AssertionError("windowing should not be planned for MPS")),
            ), mock.patch.object(
                inject_tool.Hardware_Utils, "get_available_devices", return_value=[mps_cand]
            ), mock.patch.object(
                engine, "resolve_host_cache_bytes", return_value=int(total_bytes * 0.4)
            ), redirect_stdout(out_buf), redirect_stderr(io.StringIO()):
                inject_tool.run_injection()
            self.assertNotIn("Activating partial-RAM sliding window", out_buf.getvalue())

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

    def test_tiny_prefilter_n_lt_2_bounds_check_and_unsorted_required_keys(self):
        """Medium #6 & Low #12: n=0/1 prefilter_pair_keys, bounds check in _filter_batch_by_keys, unsorted compile."""
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, [50], dim=16, dtype=np.float16)
            for n_test in (0, 1):
                for force_streamed in (False, True):
                    with self.subTest(n=n_test, force_streamed=force_streamed), mock.patch.multiple(
                        align_tool, FULL_INPUT_HDF5=path, PREFILTER_STRENGTH=50
                    ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        keys, cut = align_tool.prefilter_pair_keys(
                            headers[:n_test], [50][:n_test], force_streamed=force_streamed
                        )
                    self.assertEqual(keys.dtype, np.int64)
                    self.assertEqual(keys.size, 0)
                    self.assertEqual(cut.kept, 0)
                    self.assertEqual(cut.skipped, 0)

        # _filter_batch_by_keys rejects negative, out-of-bounds, and self-loop indices
        # even when min(a, b) * n + max(a, b) would otherwise alias a valid key!
        n = 10
        req_keys = np.array([2, 13, 24], dtype=np.int64)  # (0, 2), (1, 3), (2, 4)
        seen = np.zeros(req_keys.size, dtype=bool)
        # (-1, 12) -> -1*10 + 12 = 2 (aliases (0, 2) without bounds check!)
        # (0, 13)  -> 0*10 + 13 = 13 (aliases (1, 3) without bounds check!)
        # (2, 2)   -> self-loop
        # (1, 3)   -> valid!
        # (3, 1)   -> duplicate of (1, 3), should be skipped
        arr_i = np.array([-1, 0, 2, 1, 3, 10, 2], dtype=np.int64)
        arr_j = np.array([12, 13, 2, 3, 1, 12, 4], dtype=np.int64)
        kept_idx = align_tool._filter_batch_by_keys(arr_i, arr_j, req_keys, seen, n)
        np.testing.assert_array_equal(kept_idx, np.array([3, 6], dtype=np.int64))
        np.testing.assert_array_equal(seen, np.array([False, True, True]))

        # Medium #2: WindowedPairSession frees _pending_keys after _prepare_key_schedule()
        with tempfile.TemporaryDirectory() as tmp:
            path2, headers2 = self._make_h5(tmp, [60] * 10, dim=32, dtype=np.float16)
            store2 = self._store_with_budget(path2, headers2, 20_000)
            a_b, s_b = engine.plan_windowed_budgets(store2)
            pkeys = np.array([0 * 10 + 1, 1 * 10 + 2, 2 * 10 + 5], dtype=np.int64)
            sess = engine.WindowedPairSession.from_pending_keys(
                store2, headers2, pkeys, anchor_bytes=a_b, slab_bytes=s_b, batch_size=10
            )
            self.assertIsNone(sess._pending_keys)

        # Low #12: compile_final_output sorts unsorted required_keys before searchsorted
        with tempfile.TemporaryDirectory() as tmp:
            res_dir = os.path.join(tmp, "temp")
            os.makedirs(res_dir)
            out_net = os.path.join(tmp, "out.h5")
            with h5py.File(os.path.join(res_dir, "batch_00000.h5"), "w") as hf:
                hf.create_dataset("i", data=np.array([0, 1, 2], dtype=np.uint16))
                hf.create_dataset("j", data=np.array([1, 2, 3], dtype=np.uint16))
                hf.create_dataset("l_score", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
                hf.create_dataset("l_len", data=np.array([5, 6, 7], dtype=np.uint16))
                hf.create_dataset("g_score", data=np.array([4.0, 5.0, 6.0], dtype=np.float32))
                hf.create_dataset("g_len", data=np.array([8, 9, 10], dtype=np.uint16))
            unsorted_keys = np.array([2 * 4 + 3, 0 * 4 + 1], dtype=np.int64)
            with mock.patch.multiple(
                align_tool, RESULTS_DIR=res_dir, FINAL_OUTPUT_NET=out_net
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                align_tool.compile_final_output(
                    ["a", "b", "c", "d"], [10, 10, 10, 10], 4, "chk",
                    "esm2_t6_8m", "float16", [-2.0, 0.0],
                    required_keys=unsorted_keys,
                )
            rows = self._read_network_tuples(out_net)
            self.assertEqual(sorted((r[0], r[1]) for r in rows), [(0, 1), (2, 3)])

    def test_tiny_first_batch_tops_up_benchmark_sample_without_altering_batches(self):
        """Medium #3: tiny first_batch on resumed/sparse run tops up benchmark sample cleanly."""
        rng = np.random.default_rng(7070)
        n = 24
        headers = [f"b{i:02d}" for i in range(n)]
        arrays = [rng.standard_normal((80, 64)).astype(np.float16) for _ in range(n)]
        total_bytes = sum(a.nbytes for a in arrays)
        cuda_cand = Hardware_Utils.DeviceCandidate("cuda:0", "CUDA", torch.device("cuda:0"), "cuda")
        tiled_plan = Hardware_Utils.BenchmarkResult(cuda_cand, 1000.0, lanes=1, variant="tiled")

        # Craft sparse keys so Anchor 0 has only 2 diagonal pairs while total pending is 90 pairs.
        keys_list = [0 * n + 1, 0 * n + 2]
        for u in range(10, n):
            for v in range(u + 1, n):
                keys_list.append(u * n + v)
        req_keys = np.sort(np.array(keys_list, dtype=np.int64))
        self.assertGreater(req_keys.size, 80)

        bench_seen_counts = []
        dispatched_pairs = []

        def fake_bench(tasks, *args, **kwargs):
            bench_seen_counts.append(len(tasks))
            return [tiled_plan]

        def fake_run_batch(ranked_plans, active_idx, batch, *args, **kwargs):
            dispatched_pairs.extend((u, v) for u, v, _, _ in batch)
            return active_idx

        with tempfile.TemporaryDirectory() as tmp:
            embed_dir = os.path.join(tmp, "embed")
            net_dir = os.path.join(tmp, "net")
            os.makedirs(embed_dir)
            os.makedirs(net_dir)
            emb_name = "tiny_first_[esm2_t6_8m]_embeddings.h5"
            self._make_complete_h5(os.path.join(embed_dir, emb_name), headers, arrays)
            # Force first yielded batch from WindowedPairSession to have only 2 pairs.
            orig_from_keys = engine.WindowedPairSession.from_pending_keys

            def tiny_first_session(*args, **kwargs):
                sess = orig_from_keys(*args, **kwargs)
                orig_iter = sess.iter_batches

                def wrapped_iter():
                    all_b = list(orig_iter())
                    flat = [p for b in all_b for p in b]
                    yield flat[:2]
                    for start in range(2, len(flat), 40):
                        yield flat[start:start + 40]

                sess.iter_batches = wrapped_iter
                return sess

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
                PREFILTER_STRENGTH=50,
                BATCH_SIZE=80,
                WORKERS=2,
                HOST_CACHE_GB="auto",
                DEVICE_SELECTION="cuda:0",
                EXECUTION_MODE="tiled",
                ACCELERATOR_PRECISION="ieee_fp32",
                LOCAL_GAP_P=-2.0,
                GLOBAL_GAP_P=0.0,
                set_start_method=mock.DEFAULT,
                _validate_execution_mode_hardware=mock.Mock(),
                _execution_variants=mock.Mock(return_value=["tiled"]),
                prefilter_pair_keys=mock.Mock(return_value=(req_keys, align_tool.PrefilterCut(0.5, int(req_keys.size), 0))),
                _benchmark_processing_plans=mock.Mock(side_effect=fake_bench),
                _run_batch_with_ranked_plans=mock.Mock(side_effect=fake_run_batch),
                compile_final_output=mock.Mock(),
            ), mock.patch.object(
                align_tool.Hardware_Utils, "get_available_devices", return_value=[cuda_cand]
            ), mock.patch.object(
                align_tool.WindowedPairSession, "from_pending_keys", side_effect=tiny_first_session
            ), mock.patch.object(
                engine, "resolve_host_cache_bytes", return_value=int(total_bytes * 0.35)
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                align_tool.run_job_distributor()

        self.assertEqual(bench_seen_counts, [80])
        self.assertEqual(len(dispatched_pairs), req_keys.size)
        self.assertEqual(
            sorted(u * n + v for u, v in dispatched_pairs),
            req_keys.tolist(),
        )

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

    def test_plan_windowed_budgets_skips_mps_and_unified_memory(self):
        """HIGH #1: unified-memory MPS never allocates a host window alongside its GPU tile cache."""
        gib = engine.GIB
        lengths = [100] * 20
        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=64, dtype=np.float16)
            # Each seq = 12,800 B; total = 256,000 B; budget = 102,400 B (40%).
            store = self._store_with_budget(path, headers, 102_400)
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

    def test_non_fused_tiled_pipelines_skip_h5_open_when_all_cached(self):
        """MEDIUM #1: sync (MPS) and async (XPU/non-fused CUDA) tiled pipelines skip h5py.File when RAM-resident."""

        class FakeEvent:
            def record(self, stream):
                return None

            def query(self):
                return True

            def synchronize(self):
                return None

        class FakeAsyncBackend:
            device_type = "xpu"
            supports_async_streams = True

            def supports_tiled(self, require_memory=True):
                return True, "mock XPU"

            def create_stream(self):
                return object()

            def stream_context(self, stream):
                return mock.MagicMock()

            def create_event(self):
                return FakeEvent()

            def empty_cache(self):
                return None

        class FakeSyncMpsBackend:
            device_type = "mps"
            supports_async_streams = False

            def supports_tiled(self, require_memory=True):
                return True, "mock MPS"

            def synchronize(self):
                return None

            def empty_cache(self):
                return None

        def fake_normalize(array, device, precision="float32"):
            tensor = torch.as_tensor(array, dtype=torch.float32)
            return torch.nn.functional.normalize(tensor, p=2, dim=-1)

        def align_cb(args):
            row, col, matrix = args
            return (
                int(row),
                int(col),
                float(matrix.sum()),
                int(matrix.shape[0]),
                float(matrix.max()),
                int(matrix.shape[1]),
            )

        lengths = [16, 20, 18, 22, 14, 24]
        tasks = [(0, 1, "s000", "s001"), (0, 2, "s000", "s002"), (1, 2, "s001", "s002")]
        plan = engine.AcceleratorMemoryPlan(
            free_bytes=1 << 30,
            total_bytes=1 << 30,
            usable_bytes=1 << 30,
            tile_cache_bytes=1 << 20,
            matrix_pool_bytes=1 << 20,
            matrix_bytes=1 << 19,
            reserve_bytes=0,
            lanes=1,
            inflight_slots=2,
        )
        real_h5_file = h5py.File

        with tempfile.TemporaryDirectory() as tmp:
            path, headers = self._make_h5(tmp, lengths, dim=16, dtype=np.float16, seed=123)
            total_bytes = sum(L * 16 * 2 for L in lengths)

            def run_mode(store, backend, device, run_tasks):
                with mock.patch.object(
                    engine, "get_accelerator_backend", return_value=backend
                ), mock.patch.object(
                    engine, "_to_normalized_accelerator", side_effect=fake_normalize
                ), mock.patch.object(
                    engine.h5py, "File", side_effect=real_h5_file
                ) as h5_open:
                    res = engine.run_tiled_accelerator_pipeline(
                        run_tasks,
                        store=store,
                        lengths=lengths,
                        device=device,
                        workers=2,
                        lanes=1,
                        alignment_callback=align_cb,
                        memory_plan_override=plan,
                    )
                return sorted(res), h5_open.call_count

            for label, backend, device in (
                ("sync_mps", FakeSyncMpsBackend(), torch.device("mps")),
                ("async_xpu", FakeAsyncBackend(), torch.device("xpu:0")),
            ):
                with self.subTest(backend=label):
                    # Baseline uncached store -> opens HDF5 once per pipeline call.
                    uncached = self._store_with_budget(path, headers, 0)
                    ref_res, open_uncached = run_mode(uncached, backend, device, tasks)
                    self.assertEqual(open_uncached, 1)
                    self.assertEqual(len(ref_res), len(tasks))

                    # 1) Fully RAM-cached store -> 0 h5py.File opens, exact same results.
                    full_store = self._store_with_budget(path, headers, total_bytes * 4)
                    self.assertTrue(full_store.fully_cached)
                    reads_before = full_store.disk_reads
                    full_res, open_full = run_mode(full_store, backend, device, tasks)
                    self.assertEqual(open_full, 0)
                    self.assertEqual(full_store.disk_reads, reads_before)
                    self.assertEqual(full_res, ref_res)

                    # 2) Partial-RAM windowed store with all batch indices in anchor+slab -> 0 opens.
                    win_store = self._store_with_budget(path, headers, int(total_bytes * 0.5))
                    self.assertFalse(win_store.fully_cached)
                    self.assertTrue(win_store.open_window(anchor_rows_cap=40, slab_rows_cap=40))
                    try:
                        win_store.load_anchor([0, 1])
                        win_store.prefetch_slab([2])
                        win_store.activate_prefetched_slab()
                        self.assertTrue(win_store.all_cached({0, 1, 2}))
                        reads_before = win_store.disk_reads
                        win_res, open_win = run_mode(win_store, backend, device, tasks)
                        self.assertEqual(open_win, 0)
                        self.assertEqual(win_store.disk_reads, reads_before)
                        self.assertEqual(win_res, ref_res)

                        # 3) Partial miss (index 5 not in anchor or slab) -> opens HDF5 once cleanly.
                        miss_tasks = [(0, 5, "s000", "s005")]
                        self.assertFalse(win_store.all_cached({0, 5}))
                        miss_res, open_miss = run_mode(win_store, backend, device, miss_tasks)
                        self.assertEqual(open_miss, 1)
                        self.assertEqual(len(miss_res), 1)
                        self.assertEqual(miss_res[0][:2], (0, 5))
                    finally:
                        win_store.clear_window()


if __name__ == "__main__":
    unittest.main()
