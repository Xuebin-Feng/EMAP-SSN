# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Track 3 MSA algorithmic optimizations:
  - Batched row-parallel consensus accumulator (UPGMA + NJ bit-exactness)
  - Compact shrinking (N, N) unit-stride Neighbor-Joining (bit-exactness)
  - 2-row bit-packed 1 B/cell affine profile DP (bit-exactness)
  - Deferred lazy tree-replay FASTA materialization (byte-exactness)
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

from tools import Embedding_MSA as msa


class Track3MSAOptimizationTests(unittest.TestCase):
    def test_batched_consensus_upgma_and_nj_match_serial_bit_for_bit(self):
        rng = np.random.default_rng(20261123)
        num_seqs = 65
        m = num_seqs * (num_seqs - 1) // 2
        trees = [sch.linkage(rng.random(m), method="average") for _ in range(12)]

        # UPGMA
        ref = np.zeros(m, dtype=np.float32)
        for Z in trees:
            ref += sch.cophenet(Z).astype(np.float32)
        batched = np.zeros(m, dtype=np.float32)
        msa.add_replicates_batch_to_consensus([(Z, None) for Z in trees], num_seqs, batched)
        np.testing.assert_array_equal(ref.view(np.uint32), batched.view(np.uint32))

        # NJ path lengths: single-tree batches in order vs multi-tree batch
        nj_reps = []
        ref_nj = np.zeros(m, dtype=np.float32)
        for _ in range(6):
            dc = rng.random(m).astype(np.float64)
            Z, bl = msa.neighbor_joining_condensed(dc, num_seqs, return_branch_lengths=True)
            nj_reps.append((Z, bl))
            msa.add_replicates_batch_to_consensus([(Z, bl)], num_seqs, ref_nj)
        batched_nj = np.zeros(m, dtype=np.float32)
        msa.add_replicates_batch_to_consensus(nj_reps, num_seqs, batched_nj)
        np.testing.assert_array_equal(ref_nj.view(np.uint32), batched_nj.view(np.uint32))

    def test_compact_nj_clustered_and_ties_deterministic_across_threads(self):
        rng = np.random.default_rng(77)
        pts = rng.standard_normal((120, 8))
        centers = rng.standard_normal((6, 8)) * 3.0
        pts += centers[rng.integers(0, 6, size=120)]
        dc = pdist(pts).astype(np.float64)

        z1, bl1 = msa.neighbor_joining_condensed(dc, 120, threads=1, return_branch_lengths=True)
        z4, bl4 = msa.neighbor_joining_condensed(dc, 120, threads=4, return_branch_lengths=True)
        np.testing.assert_array_equal(z1, z4)
        np.testing.assert_array_equal(bl1, bl4)

    def test_lazy_profile_merge_and_tree_replay_match_eager_merge_byte_for_byte(self):
        rng = np.random.default_rng(1234)
        n_seq = 16
        dim = 32
        alphabet = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
        headers = [f"seq_{i}" for i in range(n_seq)]
        lengths = rng.integers(25, 65, size=n_seq).tolist()
        seqs = {
            h: rng.choice(alphabet, size=L).tobytes().decode("ascii")
            for h, L in zip(headers, lengths)
        }
        embs = [
            msa._normalize_residue_embeddings(rng.standard_normal((L, dim)).astype(np.float32))
            for L in lengths
        ]
        Z = sch.linkage(rng.random(n_seq * (n_seq - 1) // 2), method="average")

        # Self-contained eager string/profile reference (no dependency on retired merge_clusters)
        def eager_merge_ref(seqs_a, ids_a, emb_a, seqs_b, ids_b, emb_b, path):
            moves = path[::-1]
            out_a = ["" for _ in seqs_a]
            out_b = ["" for _ in seqs_b]
            vecs = []
            ia = ib = 0
            wa, wb = float(len(ids_a)), float(len(ids_b))
            tw = wa + wb
            for mv in moves:
                if mv == 1:
                    for i, s in enumerate(seqs_a):
                        out_a[i] += s[ia]
                    for i, s in enumerate(seqs_b):
                        out_b[i] += s[ib]
                    vecs.append((emb_a[ia].astype(np.float32) * wa + emb_b[ib].astype(np.float32) * wb) / tw)
                    ia += 1
                    ib += 1
                elif mv == 2:
                    for i, s in enumerate(seqs_a):
                        out_a[i] += s[ia]
                    for i, s in enumerate(seqs_b):
                        out_b[i] += "-"
                    vecs.append((emb_a[ia].astype(np.float32) * wa) / tw)
                    ia += 1
                elif mv == 3:
                    for i, s in enumerate(seqs_a):
                        out_a[i] += "-"
                    for i, s in enumerate(seqs_b):
                        out_b[i] += s[ib]
                    vecs.append((emb_b[ib].astype(np.float32) * wb) / tw)
                    ib += 1
            return out_a + out_b, ids_a + ids_b, np.stack(vecs, axis=0).astype(np.float16)

        ref_clusters = {i: ([seqs[headers[i]]], [i], embs[i]) for i in range(n_seq)}
        for step, link in enumerate(Z):
            a, b = int(link[0]), int(link[1])
            sa, ia, ea = ref_clusters.pop(a)
            sb, ib, eb = ref_clusters.pop(b)
            score = msa.compute_score_matrix_torch(ea, eb, "cpu")
            path = msa.run_global_traceback(score, -1.0, -0.1)
            ref_clusters[n_seq + step] = eager_merge_ref(sa, ia, ea, sb, ib, eb, path)
        ref_seqs, ref_ids, ref_root_emb = ref_clusters[2 * n_seq - 2]
        expected_fasta = "".join(
            f">{headers[idx]}\n{row}\n" for idx, row in zip(ref_ids, ref_seqs)
        ).encode("utf-8")

        # Lazy tree-replay path (both sequential and heavy-first DFS order)
        for order_name, merge_order in (
            ("sequential", list(range(len(Z)))),
            ("heavy_first", msa.heavy_first_merge_order(Z, n_seq)),
        ):
            with self.subTest(order=order_name):
                leaves_lazy = msa.encode_leaf_alignments(seqs, headers)
                active_profiles = {i: embs[i] for i in range(n_seq)}
                active_widths = {i: leaves_lazy[i].shape[0] for i in range(n_seq)}
                active_counts = {i: 1 for i in range(n_seq)}
                merge_children = [None] * len(Z)
                merge_moves = [None] * len(Z)
                for step in merge_order:
                    link = Z[step]
                    a, b = int(link[0]), int(link[1])
                    ea, eb = active_profiles.pop(a), active_profiles.pop(b)
                    wa, wb = active_widths.pop(a), active_widths.pop(b)
                    ca, cb = active_counts.pop(a), active_counts.pop(b)
                    score = msa.compute_score_matrix_torch(ea, eb, "cpu")
                    path = msa.run_global_traceback(score, -1.0, -0.1)
                    prof, moves = msa.merge_profiles_lazy(wa, ca, wb, cb, path, ea, eb)
                    nid = n_seq + step
                    active_profiles[nid] = prof
                    active_widths[nid] = moves.shape[0]
                    active_counts[nid] = ca + cb
                    merge_children[step] = (a, b)
                    merge_moves[step] = moves

                with tempfile.TemporaryDirectory() as td:
                    out_path = os.path.join(td, "lazy.fasta")
                    msa.write_lazy_alignment_fasta(
                        out_path,
                        2 * n_seq - 2,
                        n_seq,
                        merge_children,
                        merge_moves,
                        leaves_lazy,
                        headers,
                        active_widths[2 * n_seq - 2],
                    )
                    with open(out_path, "rb") as handle:
                        actual_fasta = handle.read()

                self.assertEqual(actual_fasta, expected_fasta)
                # Also verify root float16 profile is bit-identical!
                np.testing.assert_array_equal(
                    ref_root_emb.view(np.uint16),
                    active_profiles[2 * n_seq - 2].view(np.uint16),
                )

    def test_compact_nj_edge_cases_empty_singleton_nan_and_extreme_values(self):
        # N = 0 and N = 1 must return empty linkage/branch arrays cleanly.
        for n in (0, 1):
            z, bl = msa.neighbor_joining_compact_kernel(np.zeros((n, n), dtype=np.float64))
            self.assertEqual(z.shape, (0, 4))
            self.assertEqual(bl.shape, (0, 2))

        def assert_valid_merges(z, n):
            self.assertEqual(z.shape, (n - 1, 4))
            active = set(range(n))
            for step in range(n - 1):
                c1, c2 = int(z[step, 0]), int(z[step, 1])
                self.assertLess(c1, c2)
                self.assertIn(c1, active)
                self.assertIn(c2, active)
                active.remove(c1)
                active.remove(c2)
                active.add(n + step)
            self.assertEqual(active, {2 * n - 2})

        # N = 5 with all-NaN, partial-NaN, and extreme > 1e15 distances (no self-merge).
        d_nan5 = np.full((5, 5), np.nan, dtype=np.float64)
        z, bl = msa.neighbor_joining_compact_kernel(d_nan5.copy())
        assert_valid_merges(z, 5)
        self.assertEqual(bl.shape, (4, 2))

        d_big5 = np.full((5, 5), 1e20, dtype=np.float64)
        np.fill_diagonal(d_big5, 0.0)
        d_big5[1, 3] = d_big5[3, 1] = 1e19
        z_big5, _ = msa.neighbor_joining_compact_kernel(d_big5.copy())
        assert_valid_merges(z_big5, 5)

        # N = 80 (k >= 64): all-NaN, sparse finite (< 4 finite rows so n_top < 4!), and extreme 1e20.
        d_nan80 = np.full((80, 80), np.nan, dtype=np.float64)
        z_nan80, _ = msa.neighbor_joining_compact_kernel(d_nan80.copy())
        assert_valid_merges(z_nan80, 80)

        d_sparse_nan80 = np.full((80, 80), np.nan, dtype=np.float64)
        d_sparse_nan80[0, :] = 1.0
        d_sparse_nan80[:, 0] = 1.0
        d_sparse_nan80[1, :] = 2.0
        d_sparse_nan80[:, 1] = 2.0
        np.fill_diagonal(d_sparse_nan80, 0.0)
        z_sparse80, _ = msa.neighbor_joining_compact_kernel(d_sparse_nan80.copy())
        assert_valid_merges(z_sparse80, 80)

        d_big80 = np.full((80, 80), 1e20, dtype=np.float64)
        np.fill_diagonal(d_big80, 0.0)
        z_big80, _ = msa.neighbor_joining_compact_kernel(d_big80.copy())
        assert_valid_merges(z_big80, 80)

    def test_populate_condensed_matrix_ignores_self_loops(self):
        n = 5
        sentinel = np.float32(99.0)
        cond = np.full(n * (n - 1) // 2, sentinel, dtype=np.float32)
        edge_i = np.array([0, 0, 2, 4, 3], dtype=np.int32)
        edge_j = np.array([0, 1, 2, 4, 1], dtype=np.int32)
        edge_d = np.array([7.0, 1.5, 8.0, 9.0, 2.5], dtype=np.float32)

        msa.populate_condensed_matrix(cond, n, edge_i, edge_j, edge_d)
        expected = np.full(n * (n - 1) // 2, sentinel, dtype=np.float32)
        # (0, 1) -> idx 0; (1, 3) -> 5*1 - 1 + 3 - 1 - 1 = 5
        expected[0] = np.float32(1.5)
        expected[5] = np.float32(2.5)
        np.testing.assert_array_equal(cond, expected)
        # Specifically ensure self-loop (0, 0) did not wrap around to cond[-1].
        self.assertEqual(cond[-1], sentinel)

    def test_neighbor_joining_condensed_does_not_mutate_caller_matrix(self):
        rng = np.random.default_rng(55)
        n = 12
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                dc = rng.uniform(0.1, 2.0, n * (n - 1) // 2).astype(dtype)
                dc_orig = dc.copy()
                z, bl = msa.neighbor_joining_condensed(dc, n, return_branch_lengths=True)
                np.testing.assert_array_equal(dc, dc_orig)
                d_sq = np.ascontiguousarray(squareform(dc_orig), dtype=np.float64)
                z_ref, bl_ref = msa.neighbor_joining_compact_kernel(d_sq)
                np.testing.assert_array_equal(z, z_ref)
                np.testing.assert_array_equal(bl, bl_ref)

    def test_consensus_accumulation_uses_limited_threads(self):
        from utilities import Numba_Threads

        rng = np.random.default_rng(101)
        num_seqs = 10
        m = num_seqs * (num_seqs - 1) // 2
        dists = rng.random(m)
        z_upgma = sch.linkage(dists, method="average")
        z_nj, bl_nj = msa.neighbor_joining_condensed(dists, num_seqs, return_branch_lengths=True)
        acc_upgma = np.zeros(m, dtype=np.float32)
        acc_nj = np.zeros(m, dtype=np.float32)

        with mock.patch.object(Numba_Threads, "limited_threads", wraps=Numba_Threads.limited_threads) as spy:
            msa.add_replicates_batch_to_consensus([(z_upgma, None)], num_seqs, acc_upgma, threads=2)
            msa.add_replicates_batch_to_consensus([(z_nj, bl_nj)], num_seqs, acc_nj, threads=2)
        self.assertEqual(spy.call_count, 2)
        self.assertEqual([call.args[0] for call in spy.call_args_list], [2, 2])

    def test_msa_kernels_module_alias_and_readonly_cache_fallback(self):
        from utilities import MSA_Kernels as mk

        self.assertIs(sys.modules["MSA_Kernels"], sys.modules["utilities.MSA_Kernels"])
        calls = []

        def fake_jit(*args, **kwargs):
            calls.append(kwargs.get("cache"))
            if kwargs.get("cache"):
                def fail(fn):
                    raise RuntimeError("cannot cache function 'sample': no locator available")
                return fail
            return lambda fn: fn

        with mock.patch.object(mk, "jit", side_effect=fake_jit):
            fn = mk._compile_cached(lambda x: x + 1)
        self.assertEqual(calls, [True, False])
        self.assertEqual(fn(2), 3)


if __name__ == "__main__":
    unittest.main()
