# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Track 4 layout, spectral, and clustering algorithmic optimizations:
  - GPU fused gather+repulsion+springs+integrate+RMSD + speculative batched rollback
  - CPU exact O(N) contiguous cell-grid repulsion (bit-exact in 2D & 3D)
  - Fused parallel Gustavson MCL expand-inflate-normalize-prune & sparse_allclose
  - Parallel Numba CSR top-K UMAP k-NN builder (K <= 64 and K > 64)
  - Parallel AMG V-cycle, single-pass Galerkin contraction, and O(m^2) deflated inverse
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import scipy.sparse as sp

import Layout_Engine_SSN as ssn
import Layout_Engine_UMAP as umap_engine
from utilities import Network_Kernels as nk
from utilities import Numba_Threads
from utilities import Spectral_Start as spec

CUDA_AVAILABLE = ssn.HAS_TORCH and ssn.torch.cuda.is_available()


class Track4LayoutOptimizationTests(unittest.TestCase):
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA GPU required")
    def test_gpu_batched_stepper_and_rollback_match_single_step_bit_for_bit(self):
        import torch
        from utilities import Hardware_Acceleration as hw

        rng = np.random.default_rng(123)
        n = 600
        pos = rng.uniform(-50.0, 50.0, (n, 2)).astype(np.float32)
        e_a = np.repeat(np.arange(n, dtype=np.int32), 4)
        e_b = (e_a + rng.integers(1, 15, size=len(e_a))) % n
        edges = np.column_stack((e_a, e_b)).astype(np.int32)
        comp = np.zeros(n, dtype=np.int32)
        limits = np.full(n, 80.0, dtype=np.float32)
        active = np.ones(n, dtype=np.bool_)
        params = {
            "DT": 0.01, "DAMPING": 0.85, "SPRING_K": 3.0,
            "COULOMB_K": 10.0, "COULOMB_CUTOFF": 15.0,
            "MAX_FORCE_LIMIT": 20.0, "MAX_TOTAL_REPULSION_FORCE": 15.0,
            "MAX_STEPS": 120, "RMSD_WINDOW": 15, "RMSD_THRESHOLD": 1.0,
            "PERCENTAGE_DROP_THRESHOLD": 0.5,
        }

        # Compare single-step loop vs step_batch + snapshot/rollback
        sim_single = ssn.SSNSimulationGPU(pos.copy(), edges, comp, limits, params, active, device=torch.device("cuda:0"))
        rmsd_single = [sim_single.step(s) for s in range(37)]
        pos_single = sim_single.get_pos()

        sim_batch = ssn.SSNSimulationGPU(pos.copy(), edges, comp, limits, params, active, device=torch.device("cuda:0"))
        r_part1 = sim_batch.step_batch(0, 20)
        snap = sim_batch.snapshot_state()
        _overshot = sim_batch.step_batch(20, 32)
        sim_batch.restore_state(snap)
        r_part2 = sim_batch.step_batch(20, 17)
        pos_batch = sim_batch.get_pos()

        np.testing.assert_array_equal(pos_single, pos_batch)
        self.assertEqual(rmsd_single, r_part1 + r_part2)

        # Verify _run_layout_stage early convergence stops at the exact same step
        cand = hw.DeviceCandidate("cuda:0", "GPU", torch.device("cuda:0"), "cuda", 0, True)
        buf_batched = io.StringIO()
        with redirect_stdout(buf_batched):
            out_batched = ssn._run_layout_stage(cand, pos.copy(), edges, comp, limits, params, active)

        # Compare against manual single-step replay on the same GPU stepper
        sim_ref = ssn.SSNSimulationGPU(pos.copy(), edges, comp, limits, params, active, device=torch.device("cuda:0"))
        from collections import deque
        buf = deque(maxlen=params["RMSD_WINDOW"])
        hist = []
        stop_step = None
        for s in range(params["MAX_STEPS"]):
            r = sim_ref.step(s)
            buf.append(r)
            avg = float(np.mean(buf))
            if len(buf) == params["RMSD_WINDOW"]:
                hist.append(avg)
                if avg < params["RMSD_THRESHOLD"]:
                    stop_step = s
                    break
                if len(hist) >= params["RMSD_WINDOW"] + 10 and s > params["MAX_STEPS"] / 4.0:
                    cur_t = np.mean(hist[-10:])
                    old_t = np.mean(hist[-(params["RMSD_WINDOW"] + 10):-params["RMSD_WINDOW"]])
                    if old_t > 0:
                        drop = ((old_t - cur_t) / old_t) * 100.0
                        if drop < -ssn._RISING_RMSD_PERCENT or drop < params["PERCENTAGE_DROP_THRESHOLD"]:
                            stop_step = s
                            break
        self.assertIsNotNone(stop_step)
        self.assertIn(f"Step {stop_step}", buf_batched.getvalue())
        np.testing.assert_array_equal(out_batched, sim_ref.get_pos())

    def test_cpu_spatial_grid_matches_serial_bit_for_bit_2d_and_3d(self):
        from numba import jit
        from tests.layout_fixtures import _serial_kernel_2d, _serial_kernel_3d

        for dims, ser_fn in ((2, _serial_kernel_2d), (3, _serial_kernel_3d)):
            ser = jit(nopython=True, fastmath=True)(ser_fn)
            rng = np.random.default_rng(900 + dims)
            n = 1200
            box = 320.0 if dims == 2 else 210.0
            pos = rng.uniform(-box * 0.5, box * 0.5, (n, dims)).astype(np.float32)
            # Include a dense hotspot cluster inside the sparse layout to test hybrid node fallback!
            pos[:260] = rng.normal(0.0, 2.0, (260, dims)).astype(np.float32)
            vel = rng.normal(0.0, 1.0, (n, dims)).astype(np.float32)
            edges = np.column_stack((np.arange(n, dtype=np.int32), (np.arange(n, dtype=np.int32) + 1) % n))
            comp = np.zeros(n, dtype=np.int32)
            active = np.ones(n, dtype=np.bool_)
            active_nodes = np.arange(n, dtype=np.int32)
            limits = np.full(n, box, dtype=np.float32)
            params = {
                "DT": 0.01, "DAMPING": 0.85, "SPRING_K": 4.0,
                "COULOMB_K": 12.0, "MAX_FORCE_LIMIT": 20.0,
                "MAX_TOTAL_REPULSION_FORCE": 18.0, "COULOMB_CUTOFF": 25.0,
            }

            p_ref, v_ref = pos.copy(), vel.copy()
            r_ref = ser(p_ref, v_ref, edges, comp, active, active_nodes, limits, 0.01, 0.85, 4.0, 12.0, 20.0, 18.0, 25.0)
            sim = ssn.SSNSimulationCPU(pos.copy(), edges, comp, limits, params, active_mask=active)
            sim.vel[:] = vel
            r_opt = sim.step(0)
            np.testing.assert_array_equal(p_ref, sim.pos)
            np.testing.assert_array_equal(v_ref, sim.vel)
            self.assertEqual(r_ref, r_opt)

    def test_fused_gustavson_mcl_matches_reference_markov_clustering(self):
        from markov_clustering import mcl

        rng = np.random.default_rng(77)
        for trial in range(5):
            n = 120
            k_cl = 4
            sz = n // k_cl
            ea, eb, sc = [], [], []
            for c in range(k_cl):
                u = rng.integers(c * sz, (c + 1) * sz, 250)
                v = rng.integers(c * sz, (c + 1) * sz, 250)
                m = u < v
                ea.append(u[m]); eb.append(v[m]); sc.append(rng.uniform(0.5, 1.0, m.sum()))
            u, v = rng.integers(0, n, 40), rng.integers(0, n, 40)
            m = u < v
            ea.append(u[m]); eb.append(v[m]); sc.append(rng.uniform(0.05, 0.15, m.sum()))
            edges = np.column_stack((np.concatenate(ea), np.concatenate(eb)))
            weights = np.concatenate(sc)
            rows = np.concatenate([edges[:, 0], edges[:, 1]])
            cols = np.concatenate([edges[:, 1], edges[:, 0]])
            mat = sp.csr_matrix((np.concatenate([weights, weights]), (rows, cols)), shape=(n, n))

            for inflation in (1.5, 2.0, 3.0):
                # Reference using markov_clustering directly
                ref_m = mcl.normalize(nk._mcl_add_self_loops(mat))
                for _ in range(100):
                    last = ref_m.copy()
                    ref_m = mcl.iterate(ref_m, 2, inflation)
                    ref_m = nk._mcl_prune(ref_m, 0.001)
                    if mcl.converged(ref_m, last):
                        break
                ref_clusters = nk._mcl_get_clusters(ref_m, with_attractors=True)
                got_clusters = nk.markov_clusters_and_attractors(mat, inflation)
                self.assertEqual(got_clusters, ref_clusters)
                # Also exercise the parallel Gustavson Numba path directly on small graphs
                numba_flow = nk._converged_mcl_flow_numba(
                    mcl.normalize(nk._mcl_add_self_loops(mat)), inflation, 0.001, None
                )
                self.assertEqual(
                    nk._mcl_get_clusters(numba_flow, with_attractors=True),
                    ref_clusters,
                )

    def test_umap_csr_topk_matches_pandas_for_small_and_large_k(self):
        import pandas as pd

        rng = np.random.default_rng(314)
        n = 150
        src = rng.integers(0, n, 2500, dtype=np.int32)
        dst = rng.integers(0, n, 2500, dtype=np.int32)
        # Include exact ties and duplicate edges with differing distances
        dists = rng.choice(np.array([0.0, 0.25, 0.5, 1.0, 2.0], dtype=np.float32), size=2500)

        for k in (15, 75):
            idx_opt, dist_opt = umap_engine.build_precomputed_knn(src, dst, dists, n, k)
            # Pandas exact reference sorted by (u, d, v)
            df = pd.DataFrame({
                "u": np.concatenate([src, dst]),
                "v": np.concatenate([dst, src]),
                "d": np.concatenate([dists, dists]),
            })
            df = df[df["u"] != df["v"]].sort_values(["u", "d", "v"]).drop_duplicates(["u", "v"])
            top = df.groupby("u").head(k)
            ref_idx = np.full((n, k + 1), -1, dtype=np.int32)
            ref_dist = np.full((n, k + 1), np.inf, dtype=np.float32)
            ref_idx[:, 0] = np.arange(n)
            ref_dist[:, 0] = 0.0
            top = top.assign(rank=top.groupby("u").cumcount() + 1)
            ref_idx[top["u"].values, top["rank"].values] = top["v"].values
            ref_dist[top["u"].values, top["rank"].values] = top["d"].values

            np.testing.assert_array_equal(idx_opt, ref_idx)
            np.testing.assert_array_equal(dist_opt, ref_dist)

    def test_spectral_galerkin_and_deflated_inverse_match_reference(self):
        rng = np.random.default_rng(99)
        n = 800
        u = rng.integers(0, n, 4000, dtype=np.int32)
        v = rng.integers(0, n, 4000, dtype=np.int32)
        m = u < v
        edges = np.column_stack((u[m], v[m]))
        scores = rng.uniform(0.2, 1.0, len(edges))
        lap, rd = spec.normalized_laplacian(edges, scores, n)
        csr = lap.tocsr()

        agg, count = spec._standard_aggregation(csr.indptr, csr.indices)
        P, coarse_cand = spec._tentative_prolongator(agg, count, rd.astype(np.float64))
        R = P.T.tocsr()
        ref_coarse = (R @ csr @ P).tocsr()
        ref_coarse.sort_indices()
        got_coarse = spec._galerkin_contract(csr, P, R, agg, count)
        np.testing.assert_allclose(got_coarse.toarray(), ref_coarse.toarray(), rtol=1e-12, atol=1e-13)

        # Check O(m^2) deflated inverse against O(m^3) reference
        from scipy.linalg import pinvh
        dense = ref_coarse.toarray()
        unit = coarse_cand / np.linalg.norm(coarse_cand)
        outside = np.eye(len(unit)) - np.outer(unit, unit)
        ref_inv = outside @ pinvh(dense + np.outer(unit, unit)) @ outside
        got_inv = spec._deflated_inverse(dense, coarse_cand)
        np.testing.assert_allclose(got_inv, ref_inv, rtol=1e-12, atol=1e-12)

    def test_cpu_spatial_grid_nonfinite_and_boundary_positions_2d_and_3d(self):
        from numba import jit
        from tests.layout_fixtures import _serial_kernel_2d, _serial_kernel_3d
        from utilities import Layout_CPU_Kernels as cpu_k

        for dims, ser_fn, acc_fn in (
            (2, _serial_kernel_2d, cpu_k.accumulate_repulsion_grouped_2d),
            (3, _serial_kernel_3d, cpu_k.accumulate_repulsion_grouped_3d),
        ):
            with self.subTest(dims=dims):
                ser = jit(nopython=True, fastmath=True)(ser_fn)
                rng = np.random.default_rng(1400 + dims)
                n = 600
                cutoff = 10.0
                h = cutoff * 1.0001
                pos = rng.uniform(0.0, 10.0 * h, (n, dims)).astype(np.float32)
                # Force exact min/max boundary coordinates aligned to integer multiples of h.
                pos[0, :] = np.float32(0.0)
                pos[1, :] = np.float32(10.0 * h)
                pos[2, :] = np.nextafter(np.float32(10.0 * h), np.float32(0.0))
                vel = np.zeros((n, dims), dtype=np.float32)
                edges = np.zeros((0, 2), dtype=np.int32)
                comp = np.zeros(n, dtype=np.int32)
                active = np.ones(n, dtype=np.bool_)
                active_nodes = np.arange(n, dtype=np.int32)
                limits = np.full(n, 500.0, dtype=np.float32)
                params = {
                    "DT": 0.01, "DAMPING": 0.85, "SPRING_K": 4.0,
                    "COULOMB_K": 12.0, "MAX_FORCE_LIMIT": 20.0,
                    "MAX_TOTAL_REPULSION_FORCE": 18.0, "COULOMB_CUTOFF": cutoff,
                }

                p_ref, v_ref = pos.copy(), vel.copy()
                r_ref = ser(p_ref, v_ref, edges, comp, active, active_nodes, limits, 0.01, 0.85, 4.0, 12.0, 20.0, 18.0, cutoff)
                sim = ssn.SSNSimulationCPU(pos.copy(), edges, comp, limits, params, active_mask=active)
                r_opt = sim.step(0)
                np.testing.assert_array_equal(p_ref, sim.pos)
                np.testing.assert_array_equal(v_ref, sim.vel)
                self.assertEqual(r_ref, r_opt)

                # Non-finite coordinates at start, middle, and end of an m >= 512 run must
                # safely bypass grid construction without heap OOB or crash.
                members, run_starts, run_ends = cpu_k.group_active_nodes(comp, active_nodes)
                n_threads = Numba_Threads.numba.config.NUMBA_NUM_THREADS if cpu_k.NUMBA_AVAILABLE else 1
                for bad_idx, bad_val in ((0, np.nan), (n // 2, np.inf), (n - 1, -np.inf)):
                    pos_bad = pos.copy()
                    pos_bad[bad_idx, 0] = np.float32(bad_val)
                    acc = np.zeros_like(pos_bad)
                    rep = np.zeros_like(pos_bad)
                    run_pos = np.empty((n, dims), dtype=pos_bad.dtype)
                    acc_fn(
                        pos_bad, acc, rep, run_pos, members, run_starts, run_ends, n_threads,
                        12.0, 20.0, 18.0, cutoff,
                    )
                    # Also exercise through the fastmath=True outer physics kernel.
                    sim_bad = ssn.SSNSimulationCPU(
                        pos_bad.copy(), edges, comp, limits, params, active_mask=active
                    )
                    sim_bad.step(0)

    def test_cpu_layout_module_alias_and_readonly_cache_fallback(self):
        from utilities import Layout_CPU_Kernels as cpu_k

        self.assertIs(sys.modules["Layout_CPU_Kernels"], sys.modules["utilities.Layout_CPU_Kernels"])
        if not cpu_k.NUMBA_AVAILABLE:
            self.skipTest("Numba required")
        calls = []

        def fake_jit(*args, **kwargs):
            calls.append(kwargs.get("cache"))
            if kwargs.get("cache"):
                def fail(fn):
                    raise RuntimeError("cannot cache function 'sample': no locator available")
                return fail
            return lambda fn: fn

        with mock.patch.object(cpu_k, "jit", side_effect=fake_jit):
            fn = cpu_k._compile_cached(lambda x: x + 1)
        self.assertEqual(calls, [True, False])
        self.assertEqual(fn(2), 3)

    def test_gustavson_mcl_threshold_boundary_nan_empty_and_large_graph(self):
        from markov_clustering import mcl

        # 1. Threshold boundary (0.001) on denominators where val * (1.0 / col_sum) < 0.001 <= val / col_sum.
        for denom in (73, 84, 107):
            with self.subTest(denom=denom):
                w_small = np.float64(denom) * 0.001
                w_large = np.float64(denom) * 0.999
                # M[:, 0] = [0, 1, 0], M[:, 1] = [0, w_small, w_large], M[:, 2] = [0, 0, 0]:
                # Then (M^2)[:, 0] = [0, w_small, w_large] with exact col_sum = float(denom).
                m_test = sp.csc_matrix(
                    np.array([
                        [0.0, 0.0, 0.0],
                        [1.0, w_small, 0.0],
                        [0.0, w_large, 0.0],
                    ], dtype=np.float64)
                )
                m_test.sort_indices()
                ref_step = nk._mcl_prune(mcl.iterate(m_test, 2, 1.0), 0.001)
                with Numba_Threads.limited_threads(1):
                    acc3 = np.zeros((nk.numba.config.NUMBA_NUM_THREADS, 3), dtype=np.float64)
                    st3 = np.zeros((nk.numba.config.NUMBA_NUM_THREADS, 3), dtype=np.int32)
                    tc3 = np.empty((nk.numba.config.NUMBA_NUM_THREADS, 3), dtype=np.int32)
                    oi3 = np.empty((3, 3), dtype=np.int32)
                    ov3 = np.empty((3, 3), dtype=np.float64)
                    oc3 = np.empty(3, dtype=np.int32)
                    nk._mcl_step_batch_kernel(
                        m_test.indptr.astype(np.int64),
                        m_test.indices.astype(np.int32),
                        m_test.data.astype(np.float64),
                        0, 3, 1.0, 0.001, 1,
                        acc3, st3, tc3, oi3, ov3, oc3,
                    )
                self.assertEqual(int(oc3[0]), ref_step.getcol(0).nnz)
                self.assertEqual(oi3[0, :oc3[0]].tolist(), ref_step.getcol(0).indices.tolist())

        # 2. _mcl_sparse_allclose_kernel must return False on NaN in any branch.
        a_indptr = np.array([0, 2], dtype=np.int64)
        a_indices = np.array([0, 1], dtype=np.int32)
        b_indptr = np.array([0, 2], dtype=np.int64)
        b_indices = np.array([0, 1], dtype=np.int32)
        self.assertFalse(nk._mcl_sparse_allclose_kernel(
            a_indptr, a_indices, np.array([np.nan, 0.5], dtype=np.float64),
            b_indptr, b_indices, np.array([np.nan, 0.5], dtype=np.float64),
            1e-5, 1e-8,
        ))
        self.assertFalse(nk._mcl_sparse_allclose_kernel(
            np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int32), np.array([np.nan], dtype=np.float64),
            np.array([0, 1], dtype=np.int64), np.array([1], dtype=np.int32), np.array([0.0], dtype=np.float64),
            1e-5, 1e-8,
        ))
        self.assertFalse(nk._mcl_sparse_allclose_kernel(
            np.array([0, 1], dtype=np.int64), np.array([1], dtype=np.int32), np.array([0.0], dtype=np.float64),
            np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int32), np.array([np.nan], dtype=np.float64),
            1e-5, 1e-8,
        ))
        self.assertFalse(nk._mcl_sparse_allclose_kernel(
            np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int32), np.array([np.nan], dtype=np.float64),
            np.array([0, 0], dtype=np.int64), np.array([], dtype=np.int32), np.array([], dtype=np.float64),
            1e-5, 1e-8,
        ))
        self.assertFalse(nk._mcl_sparse_allclose_kernel(
            np.array([0, 0], dtype=np.int64), np.array([], dtype=np.int32), np.array([], dtype=np.float64),
            np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int32), np.array([np.nan], dtype=np.float64),
            1e-5, 1e-8,
        ))

        # 3. n == 0 succeeds cleanly.
        empty_csc = sp.csc_matrix((0, 0), dtype=np.float64)
        out_empty = nk._converged_mcl_flow_numba(empty_csc, 2.0, 0.001, None)
        self.assertEqual(out_empty.shape, (0, 0))
        self.assertEqual(out_empty.nnz, 0)

        # 4. Public markov_clusters_and_attractors on n = 1005 hits Numba path and matches reference.
        rng = np.random.default_rng(2026)
        n_large = 1005
        n_cl = 15
        sz = n_large // n_cl
        ea, eb, sc = [], [], []
        for c in range(n_cl):
            lo, hi = c * sz, (c + 1) * sz
            nodes = np.arange(lo, hi, dtype=np.int32)
            # Ring + chords inside each cluster for fast convergence
            ea.append(nodes); eb.append(np.roll(nodes, -1)); sc.append(np.full(sz, 1.0))
            u = rng.integers(lo, hi, sz * 3)
            v = rng.integers(lo, hi, sz * 3)
            m = u < v
            ea.append(u[m]); eb.append(v[m]); sc.append(rng.uniform(0.6, 1.0, m.sum()))
        # Sparse weak bridges between adjacent clusters
        for c in range(n_cl - 1):
            ea.append(np.array([c * sz], dtype=np.int32))
            eb.append(np.array([(c + 1) * sz], dtype=np.int32))
            sc.append(np.array([0.05]))
        u_all = np.concatenate(ea)
        v_all = np.concatenate(eb)
        w_all = np.concatenate(sc)
        mat_large = sp.csr_matrix(
            (np.concatenate([w_all, w_all]), (np.concatenate([u_all, v_all]), np.concatenate([v_all, u_all]))),
            shape=(n_large, n_large),
        )
        ref_flow = mcl.normalize(nk._mcl_add_self_loops(mat_large))
        for _ in range(100):
            last = ref_flow.copy()
            ref_flow = mcl.iterate(ref_flow, 2, 2.0)
            ref_flow = nk._mcl_prune(ref_flow, 0.001)
            if mcl.converged(ref_flow, last):
                break
        ref_res = nk._mcl_get_clusters(ref_flow, with_attractors=True)
        with mock.patch.object(nk, "_converged_mcl_flow_numba", wraps=nk._converged_mcl_flow_numba) as spy:
            got_res = nk.markov_clusters_and_attractors(mat_large, 2.0)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(got_res, ref_res)

        # 5. Negative weights on n = 1005 fall back to SciPy reference instead of Numba path.
        bad_mat = mat_large.copy()
        bad_mat.data[0] = -0.5
        with mock.patch.object(nk, "_converged_mcl_flow_numba", side_effect=AssertionError("should fall back")):
            nk._converged_mcl_flow(bad_mat, 2.0, None)

    def test_umap_csr_bounds_check_and_nonfinite_distances(self):
        # Out-of-bounds endpoints raise ValueError instead of segfaulting.
        for src, dst, n in (
            (np.array([-1], dtype=np.int32), np.array([0], dtype=np.int32), 5),
            (np.array([0], dtype=np.int32), np.array([5], dtype=np.int32), 5),
            (np.array([0], dtype=np.int32), np.array([0], dtype=np.int32), 0),
        ):
            with self.assertRaises(ValueError):
                umap_engine.build_precomputed_knn(src, dst, np.array([0.5], dtype=np.float32), n, 2)

        # Non-finite distances (NaN, +inf, -inf) must be skipped cleanly for both K <= 64 and K > 64.
        rng = np.random.default_rng(4242)
        n = 90
        src = rng.integers(0, n, 1800, dtype=np.int32)
        dst = rng.integers(0, n, 1800, dtype=np.int32)
        dists_clean = rng.uniform(0.0, 5.0, 1800).astype(np.float32)

        bad_src = np.array([0, 0, 1, 2, 3], dtype=np.int32)
        bad_dst = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        bad_d = np.array([np.nan, np.inf, -np.inf, np.nan, np.inf], dtype=np.float32)
        src_mixed = np.concatenate([bad_src, src])
        dst_mixed = np.concatenate([bad_dst, dst])
        dists_mixed = np.concatenate([bad_d, dists_clean])

        for k in (10, 70):
            idx_ref, dist_ref = umap_engine.build_precomputed_knn(src, dst, dists_clean, n, k)
            idx_got, dist_got = umap_engine.build_precomputed_knn(src_mixed, dst_mixed, dists_mixed, n, k)
            np.testing.assert_array_equal(idx_got, idx_ref)
            np.testing.assert_array_equal(dist_got, dist_ref)

    def test_spectral_start_zero_weight_row_and_prolongator_validation(self):
        import warnings

        # Zero-weight edge gives an all-zero Laplacian row: must raise ValueError cleanly with no warning.
        n_spec = 20
        ring_u = np.arange(1, n_spec, dtype=np.int32)
        ring_v = np.roll(ring_u, -1)
        edges = np.vstack([np.array([[0, 1]], dtype=np.int32), np.column_stack((ring_u, ring_v))])
        scores = np.concatenate([np.array([0.0]), np.ones(len(ring_u), dtype=np.float64)])
        lap, rd = spec.normalized_laplacian(edges, scores, n_spec)
        csr = lap.tocsr()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(ValueError):
                spec._inverse_absolute_row_sums(csr)
        runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(runtime_warns, [])

        # _lobpcg_axes(amg=True) catches the ValueError cleanly when AMG builds levels, with no warning.
        with mock.patch.object(spec, "AMG_MAX_COARSE", 4), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            axes, reason = spec._lobpcg_axes(
                csr, rd, 2, np.random.default_rng(0).standard_normal((n_spec, 3)), amg=True
            )
        runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(runtime_warns, [])
        self.assertIsNone(axes)
        self.assertIn("ValueError", reason)

        # _BlockVCycle disables _fast_levels when a prolongator does not have exact 1 nnz per row.
        a = sp.csr_matrix(np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]))
        # Row 0 has 2 entries, row 1 has 0 entries, row 2 has 1 entry (nnz == 3 == shape[0]!).
        p_bad = sp.csr_matrix(np.array([[0.6, 0.4], [0.0, 0.0], [0.0, 1.0]]))
        r_bad = p_bad.T.tocsr()
        w = spec._inverse_absolute_row_sums(a)[:, None]
        coarsest = np.eye(2, dtype=np.float64)
        cycle = spec._BlockVCycle(
            products=[spec.laplacian_operator(a)],
            weights=[w],
            prolongators=[p_bad],
            restrictions=[r_bad],
            coarsest=coarsest,
            level_csrs=[a],
        )
        self.assertIsNone(cycle._fast_levels)
        rhs = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        got = cycle(rhs)
        self.assertTrue(np.isfinite(got).all())

    def test_gpu_cutoff_zero_sort_guard(self):
        if not ssn.HAS_TORCH:
            self.skipTest("PyTorch required")
        from utilities import Layout_GPU_Kernels as lgk

        rep = object.__new__(lgk.KernelRepulsion)
        rep.dims = 2
        for bad_cutoff in (0.0, -5.0, float("nan"), float("inf")):
            rep._cutoff = bad_cutoff
            # Must return early without touching pos or dividing by zero.
            rep._sort(None)

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA GPU required")
    def test_gpu_use_skip_batched_rollback_matches_single_step_bit_for_bit(self):
        import torch

        rng = np.random.default_rng(9876)
        n = 4500  # > SKIP_MIN_NODES (4096) so use_skip=True is exercised
        pos = rng.uniform(-120.0, 120.0, (n, 2)).astype(np.float32)
        e_a = np.arange(n, dtype=np.int32)
        e_b = (e_a + 1) % n
        edges = np.column_stack((e_a, e_b)).astype(np.int32)
        comp = np.zeros(n, dtype=np.int32)
        limits = np.full(n, 200.0, dtype=np.float32)
        active = np.ones(n, dtype=np.bool_)
        params = {
            "DT": 0.01, "DAMPING": 0.85, "SPRING_K": 2.0,
            "COULOMB_K": 10.0, "COULOMB_CUTOFF": 15.0,
            "MAX_FORCE_LIMIT": 20.0, "MAX_TOTAL_REPULSION_FORCE": 15.0,
        }

        sim_single = ssn.SSNSimulationGPU(pos.copy(), edges, comp, limits, params, active, device=torch.device("cuda:0"))
        self.assertTrue(sim_single.uses_kernels)
        self.assertTrue(sim_single._repulsion.use_skip)
        rmsd_single = [sim_single.step(s) for s in range(25)]
        pos_single = sim_single.get_pos()

        sim_batch = ssn.SSNSimulationGPU(pos.copy(), edges, comp, limits, params, active, device=torch.device("cuda:0"))
        self.assertTrue(sim_batch._repulsion.use_skip)
        r_part1 = sim_batch.step_batch(0, 15)
        snap = sim_batch.snapshot_state()
        # Overshoot past RESORT_INTERVAL (20) so Morton re-sort occurs before rollback!
        _overshot = sim_batch.step_batch(15, 20)
        sim_batch.restore_state(snap)
        r_part2 = sim_batch.step_batch(15, 10)
        pos_batch = sim_batch.get_pos()

        np.testing.assert_array_equal(pos_single, pos_batch)
        self.assertEqual(rmsd_single, r_part1 + r_part2)


if __name__ == "__main__":
    unittest.main()
