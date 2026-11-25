# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ankh3 pLM plugin (src/resources/pLM_models/ankh3.py)."""

import os
import pathlib
import sys
import unittest
from unittest import mock

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for folder in (SRC, SRC / "tools", SRC / "utilities"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import BatchEncoding, T5Config, T5EncoderModel  # noqa: E402

from resources.pLM_models import ankh3  # noqa: E402
from tools import Generate_Embeddings as gen_tool  # noqa: E402
from tools.tool_helpers import Embedding_Batching as batching  # noqa: E402
from tools.tool_helpers import Model_Plugins  # noqa: E402


class _FakeAnkh3Tokenizer:
    """In-memory tokenizer mimicking Ankh3's unspaced [NLU]/[S2S] + bare-AA vocab."""

    eos_token_id = 1
    unk_token_id = 2

    def __init__(self, *, prepend_unk=False):
        self.vocab = {
            "<pad>": 0,
            "</s>": 1,
            "<unk>": 2,
            "[NLU]": 4,
            "[S2S]": 6,
        }
        for offset, ch in enumerate("ACDEFGHIKLMNPQRSTVWYXBZUO", start=8):
            self.vocab[ch] = offset
        self.prepend_unk = prepend_unk
        self.seen_inputs = []

        class _Backend:
            def __init__(self, parent):
                self._parent = parent
                self._pre_tokenizer = None

            @property
            def pre_tokenizer(self):
                return self._pre_tokenizer

            @pre_tokenizer.setter
            def pre_tokenizer(self, value):
                self._pre_tokenizer = value
                # Switching to prepend_scheme="never" stops emitting the leading <unk>.
                self._parent.prepend_unk = False

        self._tokenizer = _Backend(self)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _encode_one(self, text):
        ids = [self.unk_token_id] if self.prepend_unk else []
        rest = text
        for prefix in ("[NLU]", "[S2S]"):
            if rest.startswith(prefix):
                ids.append(self.vocab[prefix])
                rest = rest[len(prefix):]
                break
        for ch in rest:
            ids.append(self.vocab.get(ch, self.unk_token_id))
        ids.append(self.vocab["</s>"])
        return ids

    def __call__(self, texts, return_tensors=None, padding=False):
        self.seen_inputs.append(texts)
        single = isinstance(texts, str)
        rows = [self._encode_one(texts)] if single else [self._encode_one(t) for t in texts]
        if return_tensors == "pt":
            max_len = max(len(r) for r in rows) if padding else len(rows[0])
            input_ids = [r + [0] * (max_len - len(r)) for r in rows]
            attention_mask = [[1] * len(r) + [0] * (max_len - len(r)) for r in rows]
            return BatchEncoding(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            )
        return {"input_ids": rows[0] if single else rows}


class Ankh3PluginMetadataTests(unittest.TestCase):
    PLUGIN_PATH = SRC / "resources" / "pLM_models" / "ankh3.py"

    def test_static_metadata_is_complete_and_discoverable(self):
        supported, modes = Model_Plugins.read_plugin_metadata(self.PLUGIN_PATH)
        self.assertEqual(
            supported,
            ["ankh3_large", "ankh3_large_s2s", "ankh3_xl", "ankh3_xl_s2s"],
        )
        self.assertEqual(modes, {m: "local" for m in supported})

        display = Model_Plugins.read_model_display_names(self.PLUGIN_PATH)
        self.assertEqual(
            display,
            {
                "ankh3_large": "Ankh3 Large",
                "ankh3_large_s2s": "Ankh3 Large (S2S)",
                "ankh3_xl": "Ankh3 XL",
                "ankh3_xl_s2s": "Ankh3 XL (S2S)",
            },
        )

        terms = Model_Plugins.read_model_usage_terms(self.PLUGIN_PATH)
        self.assertEqual(set(terms), set(supported))
        for model_name in supported:
            self.assertEqual(terms[model_name]["license_id"], "CC-BY-NC-SA-4.0")
            self.assertTrue(terms[model_name]["requires_acknowledgement"])
            expected_repo = (
                "https://huggingface.co/ElnaggarLab/ankh3-large"
                if "large" in model_name
                else "https://huggingface.co/ElnaggarLab/ankh3-xl"
            )
            self.assertEqual(terms[model_name]["source_url"], expected_repo)

            resolved = Model_Plugins.find_model_plugin(model_name)
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.__name__, "ankh3")


class Ankh3TokenizerAlignmentTests(unittest.TestCase):
    def test_already_aligned_tokenizer_is_left_untouched(self):
        tok = _FakeAnkh3Tokenizer(prepend_unk=False)
        aligned = ankh3._residue_aligned_tokenizer(tok, "[NLU]")
        self.assertIs(aligned, tok)
        self.assertIsNone(tok._tokenizer.pre_tokenizer)

    def test_transformers5_leading_unk_prefix_is_repaired_for_both_modes(self):
        for prefix in ("[NLU]", "[S2S]"):
            with self.subTest(prefix=prefix):
                tok = _FakeAnkh3Tokenizer(prepend_unk=True)
                self.assertFalse(ankh3._is_residue_aligned(tok, prefix))
                aligned = ankh3._residue_aligned_tokenizer(tok, prefix)
                self.assertIs(aligned, tok)
                self.assertIsNotNone(tok._tokenizer.pre_tokenizer)
                self.assertTrue(ankh3._is_residue_aligned(tok, prefix))
                ids = tok(prefix + "MKTA")["input_ids"]
                self.assertEqual(ids[0], tok.convert_tokens_to_ids(prefix))
                self.assertEqual(len(ids), 6)

    def test_unrepairable_tokenizer_raises_at_load_time(self):
        class _BrokenTokenizer:
            unk_token_id = 2

            def __call__(self, text):
                return {"input_ids": [2, 4, 24, 18, 14, 8, 1]}

        with self.assertRaisesRegex(RuntimeError, "not residue-aligned"):
            ankh3._residue_aligned_tokenizer(_BrokenTokenizer(), "[NLU]")

    def test_is_residue_aligned_rejects_interior_unk_residue(self):
        class _InteriorUnkTokenizer:
            eos_token_id = 1
            unk_token_id = 2

            def convert_tokens_to_ids(self, token):
                return 4 if token == "[NLU]" else self.unk_token_id

            def __call__(self, text):
                return {"input_ids": [4, 20, self.unk_token_id, 14, 8, self.eos_token_id]}

        self.assertFalse(ankh3._is_residue_aligned(_InteriorUnkTokenizer(), "[NLU]"))

    def test_is_residue_aligned_rejects_missing_or_wrong_eos_token(self):
        class _WrongEosTokenizer:
            eos_token_id = 1
            unk_token_id = 2

            def __init__(self, trailing_id):
                self._trailing_id = trailing_id

            def convert_tokens_to_ids(self, token):
                return 4 if token == "[NLU]" else self.unk_token_id

            def __call__(self, text):
                return {"input_ids": [4, 20, 18, 14, 8, self._trailing_id]}

        for bad_trailing in (0, 2, 99):
            with self.subTest(trailing_id=bad_trailing):
                self.assertFalse(ankh3._is_residue_aligned(_WrongEosTokenizer(bad_trailing), "[NLU]"))
        self.assertTrue(ankh3._is_residue_aligned(_WrongEosTokenizer(1), "[NLU]"))


class Ankh3LoadAndInferenceTests(unittest.TestCase):
    def _tiny_encoder(self):
        torch.manual_seed(0)
        cfg = T5Config(
            vocab_size=40,
            d_model=32,
            d_kv=8,
            d_ff=64,
            num_layers=2,
            num_heads=4,
            pad_token_id=0,
            eos_token_id=1,
            feed_forward_proj="gated-silu",
        )
        return T5EncoderModel(cfg).to(dtype=torch.float32).eval()

    def test_load_model_enforces_license_gate_before_touching_hub(self):
        with mock.patch.object(
            ankh3,
            "require_model_license_acceptance",
            side_effect=Model_Plugins.ModelLicenseAcceptanceRequired("accept first"),
        ), mock.patch("transformers.T5Tokenizer.from_pretrained") as tok_loader, mock.patch(
            "transformers.T5EncoderModel.from_pretrained"
        ) as model_loader:
            with self.assertRaises(Model_Plugins.ModelLicenseAcceptanceRequired):
                ankh3.load_model("ankh3_large", torch.device("cpu"))
        tok_loader.assert_not_called()
        model_loader.assert_not_called()

    def test_load_model_uses_t5_tokenizer_fp32_and_disables_safetensors_conversion(self):
        expected = {
            "ankh3_large": ("ElnaggarLab/ankh3-large", "[NLU]"),
            "ankh3_large_s2s": ("ElnaggarLab/ankh3-large", "[S2S]"),
            "ankh3_xl": ("ElnaggarLab/ankh3-xl", "[NLU]"),
            "ankh3_xl_s2s": ("ElnaggarLab/ankh3-xl", "[S2S]"),
        }
        switch = Model_Plugins.SAFETENSORS_CONVERSION_SWITCH
        for model_name, (repo, expected_prefix) in expected.items():
            with self.subTest(model_name=model_name):
                seen = {}
                fake_tok = _FakeAnkh3Tokenizer(prepend_unk=True)
                fake_model = mock.Mock()
                fake_model.to.return_value = fake_model

                def load_encoder(hf_id, **kwargs):
                    seen["hf_id"] = hf_id
                    seen["dtype"] = kwargs.get("dtype")
                    seen["switch_during"] = os.environ.get(switch)
                    return fake_model

                with mock.patch.dict(os.environ), mock.patch.object(
                    ankh3, "require_model_license_acceptance"
                ), mock.patch(
                    "transformers.T5Tokenizer.from_pretrained", return_value=fake_tok
                ) as tok_loader, mock.patch(
                    "transformers.T5EncoderModel.from_pretrained", side_effect=load_encoder
                ):
                    os.environ.pop(switch, None)
                    tok, mod, prefix = ankh3.load_model(model_name, torch.device("cpu"))
                    switch_after = os.environ.get(switch)

                tok_loader.assert_called_once_with(repo)
                self.assertEqual(seen["hf_id"], repo)
                self.assertEqual(seen["dtype"], torch.float32)
                self.assertEqual(seen["switch_during"], "1")
                self.assertIsNone(switch_after)
                self.assertIs(tok, fake_tok)
                self.assertIs(mod, fake_model)
                self.assertEqual(prefix, expected_prefix)
                fake_model.to.assert_called_once_with(torch.device("cpu"))
                fake_model.eval.assert_called_once()

    def test_single_and_batch_slice_prefix_and_eos_and_differ_across_modes(self):
        model = self._tiny_encoder()
        tok = _FakeAnkh3Tokenizer(prepend_unk=False)
        device = torch.device("cpu")
        seqs = ["ACDEFGHIKLMNPQRSTVWY", "MKTA-BZJ"]

        nlu_obj = (tok, model, "[NLU]")
        s2s_obj = (tok, model, "[S2S]")

        # Verify exact single-vs-batch equivalence and expected residue lengths.
        for label, model_obj in (("NLU", nlu_obj), ("S2S", s2s_obj)):
            with self.subTest(mode=label):
                single = [ankh3.get_embedding(s, model_obj, device, np.float32) for s in seqs]
                batched = ankh3.get_embeddings_batch(seqs, model_obj, device, np.float32)
                self.assertEqual([a.shape for a in single], [(20, 32), (8, 32)])
                for a, b in zip(single, batched):
                    self.assertEqual(a.shape, b.shape)
                    self.assertEqual(a.dtype, np.dtype(np.float32))
                    self.assertLess(float(np.max(np.abs(a - b))), 1e-5)

        # Verify float16 storage cast preserves exact shape and dtype.
        f16 = ankh3.get_embedding("MKTA", nlu_obj, device, np.float16)
        self.assertEqual((f16.shape, f16.dtype), ((4, 32), np.dtype(np.float16)))

        # Verify [NLU] and [S2S] actually condition the encoder differently on the same weights.
        emb_nlu = ankh3.get_embedding("MKTA", nlu_obj, device, np.float32)
        emb_s2s = ankh3.get_embedding("MKTA", s2s_obj, device, np.float32)
        self.assertGreater(float(np.max(np.abs(emb_nlu - emb_s2s))), 1e-4)

        # Verify exact 1:-1 slicing (neither prefix nor </s> leaks into returned rows).
        captured = {}
        orig_forward = model.forward

        def spy_forward(**kwargs):
            out = orig_forward(**kwargs)
            captured["hidden"] = out.last_hidden_state.detach().clone()
            return out

        with mock.patch.object(model, "forward", side_effect=spy_forward):
            got = ankh3.get_embedding("MKTA", nlu_obj, device, np.float32)
        expected_slice = captured["hidden"][0, 1:5].numpy()
        np.testing.assert_allclose(got, expected_slice, atol=1e-6)

    def test_three_tuple_integrates_with_device_move_and_t5_batching_profile(self):
        model = self._tiny_encoder()
        tok = _FakeAnkh3Tokenizer(prepend_unk=False)
        model_obj = (tok, model, "[S2S]")

        moved = gen_tool._move_model_object(model_obj, torch.device("cpu"))
        self.assertEqual(len(moved), 3)
        self.assertIs(moved[0], tok)
        self.assertEqual(moved[2], "[S2S]")

        _, _, is_t5 = batching._model_memory_profile(model_obj)
        self.assertTrue(is_t5)

    def test_empty_inputs_behave_like_other_plugins(self):
        model = self._tiny_encoder()
        tok = _FakeAnkh3Tokenizer(prepend_unk=False)
        model_obj = (tok, model, "[NLU]")

        self.assertEqual(
            ankh3.get_embeddings_batch([], model_obj, torch.device("cpu"), np.float32),
            [],
        )
        with self.assertRaises(ValueError):
            ankh3.get_embedding("---***", model_obj, torch.device("cpu"), np.float32)
        with self.assertRaises(ValueError):
            ankh3.get_embeddings_batch(["---***"], model_obj, torch.device("cpu"), np.float32)


if __name__ == "__main__":
    unittest.main()
