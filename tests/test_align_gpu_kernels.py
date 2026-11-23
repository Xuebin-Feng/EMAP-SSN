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
from utilities.Similarity_Matrices import similar_pair_table


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class AlignGPUKernelsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
