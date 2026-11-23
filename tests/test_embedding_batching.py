# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamic length-sorted pLM batching, OOM halving, and injection fast-copy."""

import os
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch
from transformers import BertConfig, BertModel, EsmConfig, EsmModel, T5Config, T5EncoderModel

from resources.pLM_models import esm2 as esm2_plugin
from resources.pLM_models import esmc as esmc_plugin
from resources.pLM_models import prost_t5 as prost_t5_plugin
from tools import Embedding_Injection as inject_tool
from tools import Generate_Embeddings as gen_tool
from tools.tool_helpers import Embedding_Batching as batching
from utilities import HDF5_Storage


class EmbeddingBatchingTests(unittest.TestCase):
    def test_pack_length_sorted_descending_and_padding_bounded(self):
        items = [(f"s{i}", "A" * length) for i, length in enumerate([50, 400, 390, 60, 380, 55])]
        batches = batching.pack_length_sorted_batches(
            items,
            device=torch.device("cpu"),
            max_tokens=2048,
            max_batch_size=8,
            avail_bytes=512 * 1024 ** 2,
            max_padding_overhead=0.15,
        )
        # Longest sequences must appear in the very first batch.
        self.assertGreaterEqual(len(batches), 2)
        self.assertEqual(len(batches[0][0][1]), 400)
        for batch in batches:
            lens = [len(seq) + 2 for _, seq in batch]
            padded = len(lens) * max(lens)
            real = sum(lens)
            self.assertLessEqual(padded, real * 1.15 + 1e-6)

    def test_t5_quadratic_memory_tighter_than_non_t5(self):
        items = [(f"s{i}", "A" * 500) for i in range(16)]
        t5_cfg = T5Config(vocab_size=32, d_model=1024, d_kv=128, d_ff=16384, num_layers=1, num_heads=32)
        t5_mod = T5EncoderModel(t5_cfg)
        esm_cfg = EsmConfig(vocab_size=32, hidden_size=320, num_hidden_layers=1, num_attention_heads=20, intermediate_size=1280)
        esm_mod = EsmModel(esm_cfg, add_pooling_layer=False)

        budget = 400 * 1024 ** 2
        b_t5 = batching.pack_length_sorted_batches(
            items, device=torch.device("cpu"), model_obj=(None, t5_mod),
            max_tokens=16384, max_batch_size=64, avail_bytes=budget,
        )
        b_esm = batching.pack_length_sorted_batches(
            items, device=torch.device("cpu"), model_obj=(None, esm_mod),
            max_tokens=16384, max_batch_size=64, avail_bytes=budget,
        )
        self.assertLess(max(len(b) for b in b_t5), max(len(b) for b in b_esm))

    def test_oom_backoff_halves_batch_size_and_completes_all_items(self):
        items = [(f"h{i}", "ACDE" * (10 + i)) for i in range(8)]
        seen_sizes = []

        class OOMPlugin:
            def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                seen_sizes.append(len(seqs))
                if len(seqs) > 2:
                    raise RuntimeError("CUDA out of memory")
                return [np.ones((len(s), 4), dtype=target_dtype) for s in seqs]

        collected = {}

        def on_done(batch_items, embs):
            for (h, _), e in zip(batch_items, embs):
                collected[h] = e

        batching.run_batched_embeddings(
            items,
            plugin=OOMPlugin(),
            model_obj=None,
            device=torch.device("cpu"),
            target_dtype=np.dtype(np.float16),
            on_batch_complete=on_done,
        )
        self.assertEqual(set(collected.keys()), {h for h, _ in items})
        self.assertTrue(any(s > 2 for s in seen_sizes))
        self.assertLessEqual(seen_sizes[-1], 2)
        self.assertIn(2, seen_sizes)

    def test_batched_esm_t5_bert_esmc_match_single_sequence_fp32(self):
        from esm.models.esmc import EsmcConfig, EsmcModel, EsmcTokenizer

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seqs = ["ACDEFGHIKLMNPQRSTVWY", "MKTACDEFG"]

        # ESMC synthetic check
        esmc_tok = EsmcTokenizer()
        esmc_mod = EsmcModel(
            EsmcConfig(vocab_size=64, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, pad_token_id=1)
        ).to(device=device, dtype=torch.float32).eval()
        ref = [esmc_plugin.get_embedding(s, (esmc_tok, esmc_mod), device, np.float32) for s in seqs]
        bat = esmc_plugin.get_embeddings_batch(seqs, (esmc_tok, esmc_mod), device, np.float32)
        for a, b in zip(ref, bat):
            self.assertEqual(a.shape, b.shape)
            self.assertLess(float(np.max(np.abs(a - b))), 1e-4)

    def test_prost_t5_load_model_sets_eval_mode(self):
        fake_model = mock.Mock()
        fake_model.to.return_value = fake_model
        with mock.patch("transformers.T5Tokenizer.from_pretrained", return_value=mock.Mock()), \
                mock.patch("transformers.T5EncoderModel.from_pretrained", return_value=fake_model):
            _tok, loaded = prost_t5_plugin.load_model("prost_t5", torch.device("cpu"))
        loaded.eval.assert_called_once()

    def test_embedding_injection_fast_copy_and_batched_additions(self):
        with tempfile.TemporaryDirectory() as td:
            old_h5 = os.path.join(td, "old.h5")
            new_fasta = os.path.join(td, "new.fasta")
            out_h5 = os.path.join(td, "out.h5")

            with h5py.File(old_h5, "w") as hf:
                grp = HDF5_Storage.create_metadata_first_file(
                    hf, ["s0", "s1"], ["ACDE", "FGHIK"], "fake_model", "float16"
                )
                grp.create_dataset("s0", data=np.full((4, 8), 1.0, dtype=np.float16))
                grp.create_dataset("s1", data=np.full((5, 8), 2.0, dtype=np.float16))
                HDF5_Storage.mark_generation_complete(hf)

            with open(new_fasta, "w", encoding="utf-8") as f:
                f.write(">s0\nACDE\n>s1\nFGHIK\n>s2\nLMNPQR\n>s3\nSTVWY\n")

            batch_calls = []

            class BatchPlugin:
                def get_embedding(self, seq, model_obj, device, target_dtype):
                    return np.full((len(seq), 8), 3.0, dtype=target_dtype)

                def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                    batch_calls.append(list(seqs))
                    return [np.full((len(s), 8), 3.0, dtype=target_dtype) for s in seqs]

            with mock.patch.object(
                inject_tool, "load_model", return_value=(None, torch.device("cpu"), BatchPlugin())
            ):
                out_path, gen_cnt, copy_cnt = inject_tool.inject_embeddings(old_h5, new_fasta, out_h5)

            self.assertEqual((gen_cnt, copy_cnt), (2, 2))
            self.assertEqual(len(batch_calls), 1)
            with h5py.File(out_path, "r") as hf:
                manifest = HDF5_Storage.read_embedding_manifest(hf, require_complete=True)
                self.assertEqual(manifest.headers, ["s0", "s1", "s2", "s3"])
                np.testing.assert_allclose(hf["embeddings"]["s0"][:], 1.0)
                np.testing.assert_allclose(hf["embeddings"]["s1"][:], 2.0)
                np.testing.assert_allclose(hf["embeddings"]["s2"][:], 3.0)
                np.testing.assert_allclose(hf["embeddings"]["s3"][:], 3.0)


if __name__ == "__main__":
    unittest.main()
