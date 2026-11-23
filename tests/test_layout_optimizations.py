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
        from tests.test_parallel_kernels import _serial_kernel_2d, _serial_kernel_3d

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

            p_ref, v_ref = pos.copy(), vel.copy()
            r_ref = ser(p_ref, v_ref, edges, comp, active, active_nodes, limits, 0.01, 0.85, 4.0, 12.0, 20.0, 18.0, 25.0)
            p_opt, v_opt = pos.copy(), vel.copy()
            r_opt = ssn._get_physics_kernel(dims)(
                p_opt, v_opt, edges, comp, active, active_nodes, limits, 0.01, 0.85, 4.0, 12.0, 20.0, 18.0, 25.0
            )
            np.testing.assert_array_equal(p_ref, p_opt)
            np.testing.assert_array_equal(v_ref, v_opt)
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


if __name__ == "__main__":
    unittest.main()
