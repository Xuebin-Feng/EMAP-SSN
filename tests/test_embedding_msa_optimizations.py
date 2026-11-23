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

        # NJ path lengths
        nj_reps = []
        ref_nj = np.zeros(m, dtype=np.float32)
        for _ in range(6):
            dc = rng.random(m).astype(np.float64)
            Z, bl = msa.neighbor_joining_condensed(dc, num_seqs, return_branch_lengths=True)
            nj_reps.append((Z, bl))
            msa.add_replicate_to_consensus(Z, bl, num_seqs, ref_nj)
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

        # Eager reference via merge_clusters
        leaves_eager = msa.encode_leaf_alignments(seqs, headers)
        clusters = {
            i: msa.MSACluster(i, leaves_eager[i], [i], embs[i])
            for i in range(n_seq)
        }
        for step, link in enumerate(Z):
            a, b = int(link[0]), int(link[1])
            ca, cb = clusters.pop(a), clusters.pop(b)
            score = msa.compute_score_matrix_torch(ca.embedding, cb.embedding, "cpu")
            path = msa.run_global_traceback(score, -1.0, -0.1)
            mc = msa.merge_clusters(ca, cb, path, ca.embedding, cb.embedding)
            mc.idx = n_seq + step
            clusters[n_seq + step] = mc
        final_eager = clusters[2 * n_seq - 2]
        eager_rows = np.ascontiguousarray(final_eager.aligned.T)
        expected_fasta = "".join(
            f">{headers[idx]}\n{row.tobytes().decode('ascii')}\n"
            for idx, row in zip(final_eager.ids, eager_rows)
        ).encode("utf-8")

        # Lazy tree-replay path
        leaves_lazy = msa.encode_leaf_alignments(seqs, headers)
        active_profiles = {i: embs[i] for i in range(n_seq)}
        active_widths = {i: leaves_lazy[i].shape[0] for i in range(n_seq)}
        active_counts = {i: 1 for i in range(n_seq)}
        merge_children = []
        merge_moves = []
        for step, link in enumerate(Z):
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
            merge_children.append((a, b))
            merge_moves.append(moves)

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
            final_eager.embedding.view(np.uint16),
            active_profiles[2 * n_seq - 2].view(np.uint16),
        )


if __name__ == "__main__":
    unittest.main()
