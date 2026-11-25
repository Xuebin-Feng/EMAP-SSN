# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamic length-sorted pLM batching, OOM halving, and injection fast-copy."""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import h5py
import numpy as np
import torch
from transformers import (
    BatchEncoding,
    BertConfig,
    BertModel,
    EsmConfig,
    EsmModel,
    T5Config,
    T5EncoderModel,
)

from resources.pLM_models import ankh as ankh_plugin
from resources.pLM_models import ankh3 as ankh3_plugin
from resources.pLM_models import esm2 as esm2_plugin
from resources.pLM_models import esmc as esmc_plugin
from resources.pLM_models import prost_t5 as prost_t5_plugin
from resources.pLM_models import prot_bert as prot_bert_plugin
from resources.pLM_models import prot_t5 as prot_t5_plugin
from tools import Embedding_Injection as inject_tool
from tools import Generate_Embeddings as gen_tool
from tools.tool_helpers import Embedding_Batching as batching
from utilities import GPU_Memory_View, HDF5_Storage


class _SyntheticTokenizer:
    """Minimal in-memory tokenizer matching HF BatchEncoding contract with zero downloads."""

    _VOCAB = {ch: idx + 5 for idx, ch in enumerate("ACDEFGHIKLMNPQRSTVWYXBZUO.-")}

    def __init__(self, *, spaced=False, bos_id=None, eos_id=2, pad_id=0):
        self.spaced = spaced
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.vocab = dict(self._VOCAB)
        self.vocab["<AA2fold>"] = 4
        self.vocab["[NLU]"] = 35
        self.vocab["[S2S]"] = 36

    def _encode_one(self, text):
        if self.spaced:
            tokens = text.split()
        else:
            tokens = []
            for prefix in ("[NLU]", "[S2S]"):
                if text.startswith(prefix):
                    tokens.append(prefix)
                    text = text[len(prefix):]
                    break
            tokens.extend(list(text))
        ids = []
        if self.bos_id is not None:
            ids.append(self.bos_id)
        for tok in tokens:
            ids.append(self.vocab.get(tok, 3))
        if self.eos_id is not None:
            ids.append(self.eos_id)
        return ids

    def __call__(self, texts, return_tensors="pt", padding=False):
        if isinstance(texts, str):
            batch = [self._encode_one(texts)]
        else:
            batch = [self._encode_one(t) for t in texts]
        max_len = max(len(row) for row in batch) if padding else len(batch[0])
        input_ids = []
        attention_mask = []
        for row in batch:
            pad_n = max_len - len(row)
            input_ids.append(row + [self.pad_id] * pad_n)
            attention_mask.append([1] * len(row) + [0] * pad_n)
        return BatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )


# The real reader of the card's memory, kept before any test replaces it.
_REAL_MEMORY_READER = GPU_Memory_View.device_memory_in_use
_REAL_POOL_LIMIT = batching._pool_limit_bytes
_MIB = 1024 ** 2


class _NoRealNvidiaSmi:
    """No test starts the real nvidia-smi or reaches real torch.cuda memory calls through the pool limit.

    The reader answers "unknown" and the pool limit answers None unless a test sets a value
    itself; the cache and the "program is missing" flag are reset before and after each test.
    """

    def setUp(self):
        super().setUp()
        self._reset_memory_view_state()
        self.addCleanup(self._reset_memory_view_state)
        for patcher in (
            mock.patch.object(GPU_Memory_View, "device_memory_in_use", return_value=None),
            mock.patch.object(batching, "_pool_limit_bytes", return_value=None),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _reset_memory_view_state():
        GPU_Memory_View._OTHER_PROGRAMS_CACHE.clear()
        GPU_Memory_View._NVIDIA_SMI_MISSING = False


class EmbeddingBatchingTests(_NoRealNvidiaSmi, unittest.TestCase):
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

        torch.manual_seed(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seqs = ["ACDEFGHIKLMNPQRSTVWY", "MKTACDEFG"]

        # ESMC synthetic check
        esmc_tok = EsmcTokenizer()
        esmc_mod = EsmcModel(
            EsmcConfig(vocab_size=64, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, pad_token_id=1)
        ).to(device=device, dtype=torch.float32).eval()

        # ESM-2 synthetic check (unspaced, <cls> + <eos>)
        esm2_tok = _SyntheticTokenizer(spaced=False, bos_id=1, eos_id=2, pad_id=0)
        esm2_mod = EsmModel(
            EsmConfig(
                vocab_size=40,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                pad_token_id=0,
            ),
            add_pooling_layer=False,
        ).to(device=device, dtype=torch.float32).eval()

        # ProtBERT synthetic check (spaced, [CLS] + [SEP])
        bert_tok = _SyntheticTokenizer(spaced=True, bos_id=1, eos_id=2, pad_id=0)
        bert_mod = BertModel(
            BertConfig(
                vocab_size=40,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                pad_token_id=0,
            )
        ).to(device=device, dtype=torch.float32).eval()

        # Shared tiny T5 encoder for ProtT5, ProstT5, and Ankh
        t5_mod = T5EncoderModel(
            T5Config(
                vocab_size=40,
                d_model=32,
                d_kv=8,
                d_ff=64,
                num_layers=2,
                num_heads=4,
                pad_token_id=0,
            )
        ).to(device=device, dtype=torch.float32).eval()
        prot_t5_tok = _SyntheticTokenizer(spaced=True, bos_id=None, eos_id=2, pad_id=0)
        prost_t5_tok = _SyntheticTokenizer(spaced=True, bos_id=None, eos_id=2, pad_id=0)
        ankh_tok = _SyntheticTokenizer(spaced=False, bos_id=None, eos_id=2, pad_id=0)

        cases = [
            ("esmc", esmc_plugin, (esmc_tok, esmc_mod)),
            ("esm2", esm2_plugin, (esm2_tok, esm2_mod)),
            ("prot_bert", prot_bert_plugin, (bert_tok, bert_mod)),
            ("prot_t5", prot_t5_plugin, (prot_t5_tok, t5_mod)),
            ("prost_t5", prost_t5_plugin, (prost_t5_tok, t5_mod)),
            ("ankh", ankh_plugin, (ankh_tok, t5_mod)),
            ("ankh3_nlu", ankh3_plugin, (ankh_tok, t5_mod, "[NLU]")),
            ("ankh3_s2s", ankh3_plugin, (ankh_tok, t5_mod, "[S2S]")),
        ]
        for label, plugin, model_obj in cases:
            with self.subTest(plugin=label):
                ref = [plugin.get_embedding(s, model_obj, device, np.float32) for s in seqs]
                bat = plugin.get_embeddings_batch(seqs, model_obj, device, np.float32)
                self.assertEqual(len(ref), len(bat))
                for a, b in zip(ref, bat):
                    self.assertEqual(a.shape, b.shape)
                    self.assertEqual(a.dtype, np.dtype(np.float32))
                    self.assertEqual(b.dtype, np.dtype(np.float32))
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

    def test_cpu_benchmark_avoids_full_16_seq_sweep_on_slow_models(self):
        pending = [(f"s{i}", "ACDEFGHIKL" * (10 + i)) for i in range(40)]
        calls = []

        class SlowCpuPlugin:
            def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                calls.append(len(seqs))
                return [np.ones((len(s), 8), dtype=target_dtype) for s in seqs]

        # Simulate a slow CPU model (>= 0.25 s per sequence) via perf_counter mock.
        ticks = iter([0.0, 0.5, 1.0, 1.6, 2.0, 2.6])
        cpu_cand = gen_tool.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu", 0, False
        )
        with mock.patch.object(gen_tool.time, "perf_counter", side_effect=lambda: next(ticks)):
            _model_obj, ranked = gen_tool._benchmark_embedding_devices(
                SlowCpuPlugin(),
                "fake_slow_model",
                None,
                pending,
                np.dtype(np.float16),
                8,
                [cpu_cand],
            )
        self.assertEqual(ranked[0].candidate.spec, "cpu")
        # Must probe at most 2 single sequences instead of running 18+ sequences on CPU.
        self.assertLessEqual(sum(calls), 2)

    def test_mps_skips_device_float64_storage_cast(self):
        for dev in ("mps", "xpu", "xpu:0", "cuda", "cuda:0", torch.device("cuda:0")):
            with self.subTest(device=dev):
                self.assertIs(batching._torch_storage_dtype(np.float16, device=dev), torch.float16)
                self.assertIs(batching._torch_storage_dtype(np.float32, device=dev), torch.float32)
                self.assertIsNone(batching._torch_storage_dtype(np.float64, device=dev))
        self.assertIsNone(batching._torch_storage_dtype(np.float64, device=None))
        self.assertIs(batching._torch_storage_dtype(np.float64, device="cpu"), torch.float64)
        self.assertIs(
            batching._torch_storage_dtype(np.float64, device=torch.device("cpu")),
            torch.float64,
        )

    def test_stratified_benchmark_items_handles_small_and_zero_max_items(self):
        items = [(f"s{i}", "ACDE" * (i + 1)) for i in range(6)]
        self.assertEqual(gen_tool._stratified_benchmark_items(items, max_items=0), [])
        self.assertEqual(gen_tool._stratified_benchmark_items(items, max_items=-3), [])
        self.assertEqual(gen_tool._stratified_benchmark_items([], max_items=4), [])
        ordered = sorted(items, key=lambda pair: (len(pair[1]), pair[0]))
        self.assertEqual(
            gen_tool._stratified_benchmark_items(items, max_items=1),
            [ordered[len(ordered) // 2]],
        )
        self.assertEqual(
            gen_tool._stratified_benchmark_items(items[:1], max_items=1),
            items[:1],
        )

    def test_benchmark_handles_four_or_more_identical_sequences(self):
        pending = [(f"s{i}", "ACDEFGHIKLMN") for i in range(6)]

        class FastCpuPlugin:
            def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                return [np.ones((len(s), 4), dtype=target_dtype) for s in seqs]

        cpu_cand = gen_tool.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu", 0, False
        )
        _model_obj, ranked = gen_tool._benchmark_embedding_devices(
            FastCpuPlugin(),
            "fake_model",
            None,
            pending,
            np.dtype(np.float16),
            4,
            [cpu_cand],
        )
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].candidate.spec, "cpu")

    def test_gpu_benchmark_warms_up_with_packed_batch_and_halves_on_oom(self):
        pending = [(f"s{i}", "ACDEFGHIKL" * (10 + i)) for i in range(8)]
        cpu_cand = gen_tool.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu", None, False
        )
        gpu_cand = gen_tool.Hardware_Utils.DeviceCandidate(
            "cuda:0", "Fake GPU (CUDA)", torch.device("cuda:0"), "cuda", 0, True
        )

        # Case 1: Packed budget limits batch 0 to size 1; warmup must NOT force batch size 2.
        gpu_warmup_sizes = []

        class TightBudgetPlugin:
            def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                if getattr(device, "type", "") == "cuda":
                    gpu_warmup_sizes.append(len(seqs))
                    if len(seqs) > 1:
                        raise RuntimeError("CUDA out of memory")
                return [np.ones((len(s), 4), dtype=target_dtype) for s in seqs]

        # Force pack_length_sorted_batches to pack batch_size=1 on GPU via tiny available bytes.
        with mock.patch.object(gen_tool.Hardware_Utils, "begin_gpu_trial", return_value=None), \
                mock.patch.object(gen_tool.Hardware_Utils, "synchronize_device"), \
                mock.patch.object(gen_tool.Hardware_Utils, "release_device_cache"), \
                mock.patch.object(batching, "_available_activation_bytes", return_value=1024):
            _m, ranked = gen_tool._benchmark_embedding_devices(
                TightBudgetPlugin(),
                "fake_model",
                None,
                pending,
                np.dtype(np.float16),
                4,
                [cpu_cand, gpu_cand],
            )
        self.assertTrue(any(r.candidate.spec == "cuda:0" and r.succeeded for r in ranked))
        self.assertTrue(gpu_warmup_sizes)
        self.assertEqual(gpu_warmup_sizes[0], 1)

        # Case 2: Initial packed batch > 1 hits OOM during warmup; GPU halves down to 1 and wins.
        seen_gpu_sizes = []
        clock = [0.0]

        class WarmupOomPlugin:
            def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                if getattr(device, "type", "") == "cuda":
                    seen_gpu_sizes.append(len(seqs))
                    if len(seqs) > 1:
                        raise RuntimeError("CUDA out of memory")
                    clock[0] += 0.005 * len(seqs)
                else:
                    clock[0] += 0.30 * len(seqs)
                return [np.ones((len(s), 4), dtype=target_dtype) for s in seqs]

        with mock.patch.object(gen_tool.Hardware_Utils, "begin_gpu_trial", return_value=None), \
                mock.patch.object(gen_tool.Hardware_Utils, "synchronize_device"), \
                mock.patch.object(gen_tool.Hardware_Utils, "release_device_cache"), \
                mock.patch.object(batching, "_available_activation_bytes", return_value=4 * 1024 ** 3), \
                mock.patch.object(gen_tool.time, "perf_counter", side_effect=lambda: clock[0]):
            _m, ranked = gen_tool._benchmark_embedding_devices(
                WarmupOomPlugin(),
                "fake_model",
                None,
                pending,
                np.dtype(np.float16),
                4,
                [cpu_cand, gpu_cand],
            )
        self.assertEqual(ranked[0].candidate.spec, "cuda:0")
        self.assertTrue(any(sz > 1 for sz in seen_gpu_sizes))
        self.assertIn(1, seen_gpu_sizes)

        # Case 3: Warmup succeeds on batch 0 (size 2), timed batch 1 (size 4) OOMs -> halves & succeeds.
        timed_oom_pending = (
            [(f"long{i}", "A" * 400) for i in range(2)]
            + [(f"short{i}", "A" * 100) for i in range(6)]
        )
        timed_seen_sizes = []
        clock2 = [0.0]

        class TimedOomPlugin:
            def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                if getattr(device, "type", "") == "cuda":
                    timed_seen_sizes.append(len(seqs))
                    if len(seqs) > 2:
                        raise RuntimeError("CUDA out of memory")
                    clock2[0] += 0.005 * len(seqs)
                else:
                    clock2[0] += 0.30 * len(seqs)
                return [np.ones((len(s), 4), dtype=target_dtype) for s in seqs]

        with mock.patch.object(gen_tool.Hardware_Utils, "begin_gpu_trial", return_value=None), \
                mock.patch.object(gen_tool.Hardware_Utils, "synchronize_device"), \
                mock.patch.object(gen_tool.Hardware_Utils, "release_device_cache"), \
                mock.patch.object(batching, "_available_activation_bytes", return_value=4 * 1024 ** 3), \
                mock.patch.object(gen_tool.time, "perf_counter", side_effect=lambda: clock2[0]):
            _m, ranked = gen_tool._benchmark_embedding_devices(
                TimedOomPlugin(),
                "fake_model",
                None,
                timed_oom_pending,
                np.dtype(np.float16),
                4,
                [cpu_cand, gpu_cand],
            )
        self.assertEqual(ranked[0].candidate.spec, "cuda:0")
        self.assertTrue(any(sz > 2 for sz in timed_seen_sizes))
        self.assertLessEqual(timed_seen_sizes[-1], 2)

        # Case 4: Unrecoverable OOM even at batch_size=1 disqualifies GPU and falls back to CPU.
        class AlwaysOomGpuPlugin:
            def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                if getattr(device, "type", "") == "cuda":
                    raise RuntimeError("CUDA out of memory")
                return [np.ones((len(s), 4), dtype=target_dtype) for s in seqs]

        with mock.patch.object(gen_tool.Hardware_Utils, "begin_gpu_trial", return_value=None), \
                mock.patch.object(gen_tool.Hardware_Utils, "synchronize_device"), \
                mock.patch.object(gen_tool.Hardware_Utils, "release_device_cache"), \
                mock.patch.object(batching, "_available_activation_bytes", return_value=4 * 1024 ** 3):
            _m, ranked = gen_tool._benchmark_embedding_devices(
                AlwaysOomGpuPlugin(),
                "fake_model",
                None,
                pending,
                np.dtype(np.float16),
                4,
                [cpu_cand, gpu_cand],
            )
        self.assertEqual([r.candidate.spec for r in ranked], ["cpu"])

    def test_on_batch_complete_validation_error_leaves_no_partial_hdf5_or_failover(self):
        with tempfile.TemporaryDirectory() as td:
            fasta = os.path.join(td, "in.fasta")
            out_h5 = os.path.join(td, "out.h5")
            with open(fasta, "w", encoding="utf-8") as f:
                f.write(">s0\nACDEFG\n>s1\nACDEFH\n>s2\nACDEFI\n>s3\nACDEFK\n")

            calls = []

            class BadSecondItemPlugin:
                SUPPORTED_MODELS = ["bad_model"]
                MODEL_EXECUTION_MODES = {"bad_model": "local"}

                def load_model(self, model_name, device):
                    return object()

                def get_embedding(self, seq, model_obj, device, target_dtype):
                    raise AssertionError("Should not fall back to single-sequence path")

                def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                    calls.append((str(device), len(seqs)))
                    embs = [np.ones((len(s), 4), dtype=target_dtype) for s in seqs]
                    if len(embs) >= 2:
                        embs[1] = np.full((len(seqs[1]), 4), np.nan, dtype=target_dtype)
                    return embs

            cpu_cand = gen_tool.Hardware_Utils.DeviceCandidate(
                "cpu", "CPU", torch.device("cpu"), "cpu", None, False
            )
            gpu_cand = gen_tool.Hardware_Utils.DeviceCandidate(
                "cuda:0", "Fake GPU (CUDA)", torch.device("cuda:0"), "cuda", 0, True
            )
            ranked = [
                gen_tool.Hardware_Utils.BenchmarkResult(gpu_cand, 0.1),
                gen_tool.Hardware_Utils.BenchmarkResult(cpu_cand, 1.0),
            ]
            plugin = BadSecondItemPlugin()
            with mock.patch.object(gen_tool, "DEVICE_SELECTION", "auto"), \
                    mock.patch.object(
                        gen_tool.Hardware_Utils, "get_available_devices", return_value=[cpu_cand, gpu_cand]
                    ), \
                    mock.patch.object(
                        gen_tool, "_benchmark_embedding_devices", return_value=(object(), ranked)
                    ), \
                    mock.patch.object(batching, "_available_activation_bytes", return_value=4 * 1024 ** 3):
                with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                    gen_tool.generate_embeddings(
                        fasta, out_h5, "bad_model", "float16", plugin_loader=lambda _: plugin
                    )

            # Inference ran once on cuda:0: no batch splitting, no duplicate-key mask, no CPU failover.
            self.assertEqual(calls, [("cuda:0", 4)])
            with h5py.File(out_h5, "r") as hf:
                self.assertEqual(list(hf["embeddings"].keys()), [])

            # Also verify batch length mismatch creates zero datasets.
            out_h5_mismatch = os.path.join(td, "out_mismatch.h5")

            class ShortBatchPlugin(BadSecondItemPlugin):
                def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                    return [np.ones((len(seqs[0]), 4), dtype=target_dtype)]

            with mock.patch.object(gen_tool, "DEVICE_SELECTION", "cpu"), \
                    mock.patch.object(
                        gen_tool.Hardware_Utils, "get_available_devices", return_value=[cpu_cand]
                    ):
                with self.assertRaisesRegex(ValueError, "returned 1 embeddings for a batch of 4"):
                    gen_tool.generate_embeddings(
                        fasta, out_h5_mismatch, "bad_model", "float16",
                        plugin_loader=lambda _: ShortBatchPlugin(),
                    )
            with h5py.File(out_h5_mismatch, "r") as hf:
                self.assertEqual(list(hf["embeddings"].keys()), [])

    def test_on_batch_complete_runtime_error_skips_retry_and_failover(self):
        items = [("h0", "ACDE"), ("h1", "ACDE"), ("h2", "ACDE")]
        infer_calls = []
        failover_calls = []

        class OkPlugin:
            def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                infer_calls.append(len(seqs))
                return [np.ones((len(s), 4), dtype=target_dtype) for s in seqs]

        def failing_write(_batch, _embs):
            raise RuntimeError("simulated HDF5 disk full")

        def unexpected_failover(header, err):
            failover_calls.append((header, err))
            return None, torch.device("cpu")

        with self.assertRaisesRegex(RuntimeError, "simulated HDF5 disk full"):
            batching.run_batched_embeddings(
                items,
                plugin=OkPlugin(),
                model_obj=None,
                device=torch.device("cpu"),
                target_dtype=np.dtype(np.float16),
                on_batch_complete=failing_write,
                on_single_failure=unexpected_failover,
            )
        self.assertEqual(infer_calls, [3])
        self.assertEqual(failover_calls, [])

    def test_embedding_injection_validation_failure_creates_no_partial_datasets(self):
        with tempfile.TemporaryDirectory() as td:
            old_h5 = os.path.join(td, "old.h5")
            new_fasta = os.path.join(td, "new.fasta")
            out_h5 = os.path.join(td, "out.h5")

            with h5py.File(old_h5, "w") as hf:
                grp = HDF5_Storage.create_metadata_first_file(
                    hf, ["s0"], ["ACDE"], "fake_model", "float16"
                )
                grp.create_dataset("s0", data=np.ones((4, 8), dtype=np.float16))
                HDF5_Storage.mark_generation_complete(hf)

            with open(new_fasta, "w", encoding="utf-8") as f:
                f.write(">s0\nACDE\n>s1\nFGHIK\n>s2\nLMNPQ\n")

            class BadBatchPlugin:
                def get_embedding(self, seq, model_obj, device, target_dtype):
                    return np.ones((len(seq), 8), dtype=target_dtype)

                def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                    return [
                        np.ones((len(seqs[0]), 8), dtype=target_dtype),
                        np.full((len(seqs[1]), 8), np.nan, dtype=target_dtype),
                    ]

            with mock.patch.object(
                inject_tool, "load_model", return_value=(None, torch.device("cpu"), BadBatchPlugin())
            ):
                with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                    inject_tool.inject_embeddings(old_h5, new_fasta, out_h5)

            with h5py.File(out_h5, "r") as hf:
                # Existing s0 was copied, but neither s1 nor s2 from the invalid batch was created.
                self.assertEqual(sorted(hf["embeddings"].keys()), ["s0"])

    def test_available_activation_bytes_includes_reusable_cached_memory(self):
        gib = 1024 ** 3
        total = 8 * gib
        free = int(1.5 * gib)
        reserved = int(6.5 * gib)
        allocated = int(3.0 * gib)
        expected_cuda = int(((free + (reserved - allocated)) - int(total * 0.18)) * 0.50)

        with mock.patch.object(torch.cuda, "mem_get_info", return_value=(free, total)), \
                mock.patch.object(torch.cuda, "memory_reserved", return_value=reserved), \
                mock.patch.object(torch.cuda, "memory_allocated", return_value=allocated):
            budget = batching._available_activation_bytes(torch.device("cuda:0"))
        self.assertEqual(budget, expected_cuda)

        expected_xpu = int(((free + (reserved - allocated)) - int(total * 0.20)) * 0.50)
        fake_xpu = mock.Mock()
        fake_xpu.mem_get_info.return_value = (free, total)
        fake_xpu.memory_reserved.return_value = reserved
        fake_xpu.memory_allocated.return_value = allocated
        with mock.patch.object(torch, "xpu", fake_xpu, create=True):
            xpu_budget = batching._available_activation_bytes(torch.device("xpu:0"))
        self.assertEqual(xpu_budget, expected_xpu)

    def test_benchmark_releases_gpu_cache_on_gpu_win_and_cpu_win(self):
        pending = [(f"s{i}", "ACDEFGHIKL" * (5 + i)) for i in range(6)]
        cpu_cand = gen_tool.Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu", None, False
        )
        gpu_cand = gen_tool.Hardware_Utils.DeviceCandidate(
            "cuda:0", "Fake GPU (CUDA)", torch.device("cuda:0"), "cuda", 0, True
        )

        for gpu_faster in (True, False):
            with self.subTest(gpu_faster=gpu_faster):
                events = []
                clock = [0.0]

                class TimedPlugin:
                    def get_embeddings_batch(self, seqs, model_obj, device, target_dtype):
                        dev_type = getattr(device, "type", "")
                        events.append(("infer", dev_type))
                        if dev_type == "cuda":
                            clock[0] += 0.01 if gpu_faster else 2.0
                        else:
                            clock[0] += 1.0 if gpu_faster else 0.01
                        return [np.ones((len(s), 4), dtype=target_dtype) for s in seqs]

                def record_release(cand):
                    events.append(("release", getattr(cand, "spec", str(cand))))

                with mock.patch.object(gen_tool.Hardware_Utils, "begin_gpu_trial", return_value=None), \
                        mock.patch.object(gen_tool.Hardware_Utils, "synchronize_device"), \
                        mock.patch.object(
                            gen_tool.Hardware_Utils, "release_device_cache", side_effect=record_release
                        ), \
                        mock.patch.object(batching, "_available_activation_bytes", return_value=4 * 1024 ** 3), \
                        mock.patch.object(gen_tool.time, "perf_counter", side_effect=lambda: clock[0]):
                    _m, ranked = gen_tool._benchmark_embedding_devices(
                        TimedPlugin(),
                        "fake_model",
                        None,
                        pending,
                        np.dtype(np.float16),
                        4,
                        [cpu_cand, gpu_cand],
                    )
                expected_winner = "cuda:0" if gpu_faster else "cpu"
                self.assertEqual(ranked[0].candidate.spec, expected_winner)
                last_cuda_infer = max(
                    idx for idx, ev in enumerate(events) if ev == ("infer", "cuda")
                )
                post_infer_releases = [
                    ev for idx, ev in enumerate(events)
                    if idx > last_cuda_infer and ev == ("release", "cuda:0")
                ]
                self.assertTrue(
                    post_infer_releases,
                    f"Expected release_device_cache('cuda:0') after CUDA benchmark, got {events}",
                )


_CARD_UUID = "GPU-8f6c3a2e-1b2c-4d5e-9f00-112233445566"
_OTHER_CARD_UUID = "GPU-0a1b2c3d-0000-4000-8000-aabbccddeeff"
_CARD_TOTAL_MIB = 12282


def _smi_result(*rows, returncode=0):
    """A CompletedProcess whose output has one ``uuid, used, total`` row per tuple."""
    text = "".join(f"{uuid}, {used}, {total}\n" for uuid, used, total in rows)
    return subprocess.CompletedProcess([], returncode, stdout=text, stderr="")


def _expected_budget(total, effective_free):
    """The budget formula for a CUDA card, written out as the module documents it."""
    reserve = max(512 * _MIB, int(total * 0.18))
    return max(128 * _MIB, int(max(0, effective_free - reserve) * 0.50))


class DeviceMemoryViewTests(_NoRealNvidiaSmi, unittest.TestCase):
    """The budget beside the second view of the card, the pool limit, and the pool release in the run loop."""

    @contextlib.contextmanager
    def _cuda_card(self, *, free_mib, reserved_mib, allocated_mib, run=None, torch_uuid=None):
        """Patch torch and (when ``run`` is given) the real reader on top of that mock; yield stdout."""
        total = _CARD_TOTAL_MIB * _MIB
        if torch_uuid is None:
            props = mock.Mock(side_effect=RuntimeError("no properties"))
        else:
            props = mock.Mock(return_value=types.SimpleNamespace(uuid=torch_uuid))
        buffer = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(torch.cuda, "mem_get_info", return_value=(free_mib * _MIB, total)))
            stack.enter_context(mock.patch.object(torch.cuda, "memory_reserved", return_value=reserved_mib * _MIB))
            stack.enter_context(mock.patch.object(torch.cuda, "memory_allocated", return_value=allocated_mib * _MIB))
            stack.enter_context(mock.patch.object(torch.cuda, "get_device_properties", props))
            stack.enter_context(mock.patch.object(torch.version, "cuda", "12.4"))
            if run is not None:
                # tests/__init__.py turns the switch off for the whole run; this block asks the reader.
                stack.enter_context(mock.patch.object(GPU_Memory_View, "other_programs_switch_on", return_value=True))
                stack.enter_context(mock.patch.object(GPU_Memory_View, "device_memory_in_use", _REAL_MEMORY_READER))
                stack.enter_context(mock.patch("subprocess.run", run))
            stack.enter_context(contextlib.redirect_stdout(buffer))
            yield buffer

    def _budget(self, **card):
        with self._cuda_card(**card):
            return batching._available_activation_bytes(torch.device("cuda:0"))

    @staticmethod
    def _smi(used_mib, total_mib=_CARD_TOTAL_MIB):
        return mock.Mock(return_value=_smi_result((_CARD_UUID, used_mib, total_mib)))

    # ---- the budget ignores the second view --------------------------------------------

    def test_budget_of_a_cuda_card_is_todays_value_even_when_others_hold_a_lot(self):
        free, reserved, allocated = 11106, 3000, 2000
        total = _CARD_TOTAL_MIB * _MIB
        run = self._smi(reserved + 6000)  # nvidia-smi says others hold 6,000 MiB
        with self._cuda_card(
            free_mib=free, reserved_mib=reserved, allocated_mib=allocated, run=run
        ):
            with mock.patch.object(
                GPU_Memory_View, "device_memory_in_use", side_effect=AssertionError("budget must not read it")
            ) as reader:
                budget = batching._available_activation_bytes(torch.device("cuda:0"))
        self.assertEqual(
            budget, _expected_budget(total, min(total, (free + reserved - allocated) * _MIB))
        )
        reader.assert_not_called()
        run.assert_not_called()

    # ---- the limit of torch's cached pool ---------------------------------------------

    def _pool_limit(self, *, free_mib, reserved_mib, used_mib=None, device="cuda:0", windows=True):
        run = None if used_mib is None else self._smi(used_mib)
        with self._cuda_card(
            free_mib=free_mib, reserved_mib=reserved_mib, allocated_mib=0, run=run
        ), mock.patch.object(
            GPU_Memory_View, "card_limit_is_not_enforced", return_value=windows
        ), mock.patch.object(batching, "_pool_limit_bytes", _REAL_POOL_LIMIT):
            return batching._pool_limit_bytes(torch.device(device))

    @staticmethod
    def _margin(total):
        return max(512 * _MIB, int(total * 0.05))

    def test_pool_limit_in_the_measured_windows_case(self):
        total = _CARD_TOTAL_MIB * _MIB
        pool = self._pool_limit(free_mib=6006, reserved_mib=4700, used_mib=4700 + 3268)
        self.assertIsNotNone(pool)
        self.assertEqual(pool.others, 3268 * _MIB)
        self.assertEqual(pool.total, total)
        self.assertEqual(pool.margin, self._margin(total))
        self.assertEqual(pool.limit, total - 3268 * _MIB - self._margin(total))
        self.assertAlmostEqual(pool.limit / _MIB, 8400, delta=1)

    def test_pool_limit_without_the_second_view_is_reserved_plus_free_minus_margin(self):
        total = _CARD_TOTAL_MIB * _MIB
        pool = self._pool_limit(free_mib=6006, reserved_mib=4700)
        self.assertIsNotNone(pool)
        self.assertIsNone(pool.others)
        self.assertEqual(pool.limit, (4700 + 6006) * _MIB - self._margin(total))

    def test_pool_limit_never_passes_the_card_when_cuda_reports_too_much_free(self):
        total = _CARD_TOTAL_MIB * _MIB
        # 4,700 + 11,106 = 15,806 MiB would be more than the 12,282 MiB card.
        pool = self._pool_limit(free_mib=11106, reserved_mib=4700)
        self.assertIsNotNone(pool)
        self.assertIsNone(pool.others)
        self.assertEqual(pool.limit, total - self._margin(total))

    def test_pool_limit_ignores_the_second_view_when_cuda_already_shows_less_room(self):
        # CUDA's own figure (Linux-like) is already the smaller room.
        total = _CARD_TOTAL_MIB * _MIB
        pool = self._pool_limit(free_mib=2000, reserved_mib=4700, used_mib=4700 + 100)
        self.assertIsNotNone(pool)
        self.assertEqual(pool.limit, (4700 + 2000) * _MIB - self._margin(total))

    def test_pool_limit_is_none_when_the_card_limit_is_enforced(self):
        self.assertIsNone(
            self._pool_limit(free_mib=6006, reserved_mib=4700, used_mib=7968, windows=False)
        )

    def test_pool_limit_is_none_for_cpu_xpu_and_mps_devices(self):
        for device in ("cpu", "xpu:0", "mps"):
            with self.subTest(device=device):
                self.assertIsNone(
                    self._pool_limit(free_mib=6006, reserved_mib=4700, used_mib=7968, device=device)
                )

    def test_pool_limit_is_none_when_there_is_no_room_beyond_a_gibibyte(self):
        # 4,700 + 1,000 - 614 = 5,086 MiB is under 4,700 + 1,024.
        self.assertIsNone(self._pool_limit(free_mib=1000, reserved_mib=4700))

    def test_pool_limit_is_none_without_an_exception_when_mem_get_info_raises(self):
        with self._cuda_card(free_mib=6006, reserved_mib=4700, allocated_mib=0), \
                mock.patch.object(GPU_Memory_View, "card_limit_is_not_enforced", return_value=True), \
                mock.patch.object(batching, "_pool_limit_bytes", _REAL_POOL_LIMIT), \
                mock.patch.object(torch.cuda, "mem_get_info", side_effect=RuntimeError("no")):
            self.assertIsNone(batching._pool_limit_bytes(torch.device("cuda:0")))

    # ---- the loop ---------------------------------------------------------------------

    @staticmethod
    def _long_items():
        # 24 sequences of 1,000 residues: three batches of eight under the 8,192-token cap.
        return [(f"h{i:02d}", "A" * 1000) for i in range(24)]

    @staticmethod
    def _estimate_of_a_batch():
        bytes_per_tok, bytes_per_quad, _ = batching._model_memory_profile(None)
        return 8 * 1002 * bytes_per_tok + 8 * 1002 * 1002 * bytes_per_quad

    @contextlib.contextmanager
    def _pool_run(self, *, pool, reserved_mib, allocated_mib, frees=False, limit_side_effect=None):
        """Run-loop patches: a pool limit, fake reserved/allocated memory, a recording cache release."""
        state = {"reserved": reserved_mib * _MIB, "allocated": allocated_mib * _MIB}
        releases = []

        def release(device):
            releases.append(str(device))
            if frees:
                state["reserved"] = state["allocated"]

        limit_mock = mock.Mock(return_value=pool, side_effect=limit_side_effect)
        buffer = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(batching, "_pool_limit_bytes", limit_mock))
            stack.enter_context(mock.patch.object(batching, "_available_activation_bytes", return_value=4 * 1024 ** 3))
            stack.enter_context(mock.patch.object(torch.cuda, "memory_reserved", side_effect=lambda *a: state["reserved"]))
            stack.enter_context(mock.patch.object(torch.cuda, "memory_allocated", side_effect=lambda *a: state["allocated"]))
            stack.enter_context(mock.patch.object(batching, "clear_device_cache", side_effect=release))
            stack.enter_context(contextlib.redirect_stdout(buffer))
            yield types.SimpleNamespace(state=state, releases=releases, out=buffer, limit=limit_mock)

    def _embed(self, items, device=None, plugin=None, on_single_failure=None):
        seen = []

        class OkPlugin:
            def get_embeddings_batch(self, seqs, model_obj, dev, target_dtype):
                return [np.ones((len(s), 4), dtype=target_dtype) for s in seqs]

        batching.run_batched_embeddings(
            items,
            plugin=plugin or OkPlugin(),
            model_obj=None,
            device=device if device is not None else torch.device("cuda:0"),
            target_dtype=np.dtype(np.float16),
            on_batch_complete=lambda batch, _embs: seen.append([key for key, _ in batch]),
            on_single_failure=on_single_failure,
        )
        return seen

    def test_loop_releases_the_pool_when_reserved_plus_the_batch_passes_the_limit(self):
        estimate_mib = self._estimate_of_a_batch() / _MIB  # about 1,545 MiB
        pool = batching._PoolLimit(6500 * _MIB, 600 * _MIB, None, _CARD_TOTAL_MIB * _MIB)
        self.assertGreater(6000 + estimate_mib, 6500)
        with self._pool_run(pool=pool, reserved_mib=6000, allocated_mib=3000, frees=True) as run:
            seen = self._embed(self._long_items())
        # Released before the first batch; the pool is then small enough for the rest.
        self.assertEqual(run.releases, ["cuda:0"])
        self.assertEqual(len(seen), 3)

    def test_loop_does_not_release_when_the_batch_fits_under_the_limit(self):
        pool = batching._PoolLimit(9000 * _MIB, 600 * _MIB, None, _CARD_TOTAL_MIB * _MIB)
        self.assertLess(6000 + self._estimate_of_a_batch() / _MIB, 9000)
        with self._pool_run(pool=pool, reserved_mib=6000, allocated_mib=3000) as run:
            self._embed(self._long_items())
        self.assertEqual(run.releases, [])
        self.assertEqual(run.out.getvalue(), "")

    def test_loop_does_not_release_when_less_than_64_mib_is_cached_and_unused(self):
        pool = batching._PoolLimit(6500 * _MIB, 600 * _MIB, None, _CARD_TOTAL_MIB * _MIB)
        with self._pool_run(pool=pool, reserved_mib=6000, allocated_mib=6000 - 63) as run:
            self._embed(self._long_items())
        self.assertEqual(run.releases, [])
        # Exactly 64 MiB unused is enough to release.
        with self._pool_run(pool=pool, reserved_mib=6000, allocated_mib=6000 - 64) as run:
            self._embed(self._long_items())
        self.assertTrue(run.releases)

    def test_loop_never_touches_the_pool_when_there_is_no_limit(self):
        with self._pool_run(pool=None, reserved_mib=11000, allocated_mib=1000) as run:
            seen = self._embed(self._long_items())
        self.assertEqual(len(seen), 3)
        self.assertEqual(run.releases, [])
        self.assertEqual(run.out.getvalue(), "")

    def test_loop_prints_the_release_line_once_however_many_releases_happen(self):
        pool = batching._PoolLimit(6500 * _MIB, 600 * _MIB, None, _CARD_TOTAL_MIB * _MIB)
        expected = (
            "[Hardware] GPU memory on cuda:0 is nearly full; embedding releases its unused "
            "memory so that Windows does not move it to system memory.\n"
        )
        # The release frees nothing here, so every one of the three batches releases again.
        with self._pool_run(pool=pool, reserved_mib=6000, allocated_mib=3000) as run:
            self._embed(self._long_items())
        self.assertEqual(len(run.releases), 3)
        self.assertEqual(run.out.getvalue(), expected)

    def test_loop_release_line_names_other_programs_only_from_256_mib(self):
        total = _CARD_TOTAL_MIB * _MIB
        with_others = (
            "[Hardware] GPU memory on cuda:0 is nearly full (other programs hold 3.2 GB of its "
            "12.0 GB); embedding releases its unused memory so that Windows does not move it "
            "to system memory.\n"
        )
        plain = (
            "[Hardware] GPU memory on cuda:0 is nearly full; embedding releases its unused "
            "memory so that Windows does not move it to system memory.\n"
        )
        for others, expected in ((3268 * _MIB, with_others), (256 * _MIB, None), (255 * _MIB, plain), (None, plain)):
            with self.subTest(others=others):
                pool = batching._PoolLimit(6500 * _MIB, 600 * _MIB, others, total)
                with self._pool_run(pool=pool, reserved_mib=6000, allocated_mib=3000) as run:
                    self._embed(self._long_items())
                if expected is None:  # exactly 256 MiB: the wording with the numbers
                    self.assertIn("other programs hold 0.2 GB of its 12.0 GB", run.out.getvalue())
                else:
                    self.assertEqual(run.out.getvalue(), expected)

    def test_loop_warns_once_when_the_pool_stays_past_the_limit_plus_margin(self):
        pool = batching._PoolLimit(6500 * _MIB, 600 * _MIB, None, _CARD_TOTAL_MIB * _MIB)
        expected = (
            "[Hardware] Embedding needs more GPU memory on cuda:0 than is free; Windows may "
            "move part of it to system memory, which is several times slower. Closing other "
            "programs that use the GPU helps.\n"
        )
        # Nothing is cached and unused (reserved == allocated), so there is no release line.
        with self._pool_run(pool=pool, reserved_mib=7200, allocated_mib=7200) as run:
            seen = self._embed(self._long_items())
        self.assertEqual(len(seen), 3)
        self.assertEqual(run.releases, [])
        self.assertEqual(run.out.getvalue(), expected)

    def test_loop_does_not_warn_when_the_pool_stays_within_the_limit_plus_margin(self):
        pool = batching._PoolLimit(6500 * _MIB, 600 * _MIB, None, _CARD_TOTAL_MIB * _MIB)
        with self._pool_run(pool=pool, reserved_mib=7100, allocated_mib=7100) as run:
            self._embed(self._long_items())
        self.assertEqual(run.out.getvalue(), "")

    def test_loop_embeds_the_same_batches_with_and_without_a_limit(self):
        pool = batching._PoolLimit(6500 * _MIB, 600 * _MIB, None, _CARD_TOTAL_MIB * _MIB)
        items = self._long_items() + [(f"s{i:02d}", "A" * 120) for i in range(10)]
        with self._pool_run(pool=None, reserved_mib=6000, allocated_mib=3000):
            without_limit = self._embed(items)
        with self._pool_run(pool=pool, reserved_mib=6000, allocated_mib=3000):
            with_limit = self._embed(items)
        self.assertEqual(with_limit, without_limit)
        self.assertEqual(
            sorted(key for batch in with_limit for key in batch), sorted(key for key, _ in items)
        )

    def test_loop_keeps_running_when_reading_the_pool_fails(self):
        pool = batching._PoolLimit(6500 * _MIB, 600 * _MIB, None, _CARD_TOTAL_MIB * _MIB)
        with self._pool_run(pool=pool, reserved_mib=6000, allocated_mib=3000) as run:
            with mock.patch.object(torch.cuda, "memory_reserved", side_effect=RuntimeError("driver")):
                seen = self._embed(self._long_items())
        self.assertEqual(len(seen), 3)
        self.assertEqual(run.releases, [])
        self.assertEqual(run.out.getvalue(), "")

    def test_loop_computes_the_limit_again_for_the_device_after_a_failover(self):
        pool = batching._PoolLimit(6500 * _MIB, 600 * _MIB, None, _CARD_TOTAL_MIB * _MIB)
        calls = []

        class FailsOnFirstCard:
            def get_embeddings_batch(self, seqs, model_obj, dev, target_dtype):
                calls.append(str(dev))
                if str(dev) == "cuda:0":
                    raise RuntimeError("driver exploded")
                return [np.ones((len(s), 4), dtype=target_dtype) for s in seqs]

        def failover(header, error):
            return None, torch.device("cuda:1")

        items = [("a", "A" * 50), ("b", "A" * 50)]
        with self._pool_run(pool=pool, reserved_mib=100, allocated_mib=100) as run:
            seen = self._embed(items, plugin=FailsOnFirstCard(), on_single_failure=failover)
        self.assertEqual(sorted(key for batch in seen for key in batch), ["a", "b"])
        self.assertIn("cuda:1", calls)
        self.assertEqual([str(c.args[0]) for c in run.limit.call_args_list], ["cuda:0", "cuda:1"])


if __name__ == "__main__":
    unittest.main()
