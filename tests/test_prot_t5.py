"""The prot_t5 pLM plugin: ProtT5-XL-U50 per-residue embeddings.

The first class pins the input contract with a stand-in tokenizer and model:
residues joined by spaces, one row per residue, the trailing </s> dropped. The
second loads a tiny T5 encoder-decoder saved the way a publisher might upload
it (float16) and checks the plugin keeps only the encoder, in float32 and in
eval mode. The last checks the contract against the real ProtT5 vocabulary when
its tokenizer is already in the local Hugging Face cache; nothing is
downloaded.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from resources.pLM_models import prot_t5

HF_ID = "Rostlab/prot_t5_xl_uniref50"
EOS_ID = 1
UNK_ID = 2


class SpacedTokenizer:
    """Encodes like ProtT5's tokenizer: one id per space-separated word, then </s>.

    A word that is not one residue letter encodes as <unk>, as an unspaced
    sequence would with the real vocabulary.
    """

    def __init__(self):
        self.texts = []

    def __call__(self, text, return_tensors=None):
        self.texts.append(text)
        ids = [
            ord(word) if len(word) == 1 and word.isalpha() else UNK_ID
            for word in text.split()
        ]
        input_ids = torch.tensor([ids + [EOS_ID]])
        return _Batch(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


class _Batch(dict):
    def to(self, device):
        return self


class RowPerTokenModel:
    """Returns row i = [token id, position i] for every token it is given."""

    def __call__(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32)
        rows = torch.stack([input_ids[0].to(torch.float32), positions], dim=1)
        return mock.Mock(last_hidden_state=rows.unsqueeze(0))


class EmbeddingContractTests(unittest.TestCase):
    def embed(self, sequence, dtype=np.float32):
        tokenizer = SpacedTokenizer()
        rows = prot_t5.get_embedding(
            sequence, (tokenizer, RowPerTokenModel()), "cpu", dtype
        )
        return tokenizer.texts, rows

    def test_residues_are_spaced_and_the_trailing_eos_row_is_dropped(self):
        texts, rows = self.embed("MKTA")
        self.assertEqual(texts, ["M K T A"])
        # One row per residue, in order, and none for </s>.
        self.assertEqual(rows[:, 0].tolist(), [ord(code) for code in "MKTA"])
        self.assertEqual(rows[:, 1].tolist(), [0, 1, 2, 3])

    def test_the_published_mapping_is_applied_before_tokenizing(self):
        texts, rows = self.embed("--mbzuojx.c**")
        self.assertEqual(texts, ["M X X X X X X X C"])
        self.assertEqual(len(rows), 9)
        self.assertNotIn(UNK_ID, rows[:, 0].tolist())

    def test_rows_take_the_requested_storage_precision(self):
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype):
                _, rows = self.embed("ACD", dtype)
                self.assertEqual(rows.dtype, np.dtype(dtype))

    def test_a_sequence_without_residues_is_refused_before_the_model_runs(self):
        tokenizer = SpacedTokenizer()
        with self.assertRaisesRegex(ValueError, "no supported amino-acid"):
            prot_t5.get_embedding("-.*", (tokenizer, RowPerTokenModel()), "cpu", np.float32)
        self.assertEqual(tokenizer.texts, [])


class LoadModelTests(unittest.TestCase):
    def test_only_the_encoder_is_loaded_in_float32_and_eval_mode(self):
        from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration

        # Shaped like the real checkpoint: a whole encoder-decoder, d_ff from
        # feed_forward_proj "relu", saved in float16 as a publisher might.
        config = T5Config(
            vocab_size=128, d_model=16, d_kv=4, d_ff=32, num_layers=2,
            num_decoder_layers=2, num_heads=4, feed_forward_proj="relu",
            dropout_rate=0.1, pad_token_id=0, eos_token_id=EOS_ID,
            decoder_start_token_id=0,
        )
        with tempfile.TemporaryDirectory() as temp:
            torch.manual_seed(42)
            T5ForConditionalGeneration(config).to(torch.float16).save_pretrained(temp)
            with mock.patch(
                "transformers.T5Tokenizer.from_pretrained", return_value="tokenizer"
            ) as tokenizer_loader:
                tokenizer, model = prot_t5.load_model(temp, "cpu")

        tokenizer_loader.assert_called_once_with(temp, do_lower_case=False)
        self.assertEqual(tokenizer, "tokenizer")
        self.assertIsInstance(model, T5EncoderModel)
        self.assertFalse(hasattr(model, "decoder"))
        self.assertEqual({p.dtype for p in model.parameters()}, {torch.float32})
        # Dropout is off, so the same sequence always gives the same rows.
        self.assertFalse(model.training)

        tokenizer = SpacedTokenizer()
        first = prot_t5.get_embedding("MKTAY", (tokenizer, model), "cpu", np.float32)
        second = prot_t5.get_embedding("MKTAY", (tokenizer, model), "cpu", np.float32)
        self.assertEqual(first.shape, (5, 16))
        np.testing.assert_array_equal(first, second)

    def test_the_identifier_maps_to_the_unrounded_checkpoint(self):
        with mock.patch(
            "transformers.T5Tokenizer.from_pretrained", return_value="tokenizer"
        ), mock.patch("transformers.T5EncoderModel.from_pretrained") as model_loader:
            prot_t5.load_model("prot_t5", "cpu")
        model_loader.assert_called_once_with(HF_ID, dtype=torch.float32)


def _cached_tokenizer():
    """The real ProtT5 tokenizer from the local cache, or None if it is not there."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from transformers import T5Tokenizer

        if not isinstance(try_to_load_from_cache(HF_ID, "spiece.model"), str):
            return None
        return T5Tokenizer.from_pretrained(
            HF_ID, do_lower_case=False, local_files_only=True
        )
    except Exception:
        return None


class RealVocabularyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _cached_tokenizer()
        if cls.tokenizer is None:
            raise unittest.SkipTest(f"{HF_ID} tokenizer is not in the local cache")

    def ids(self, text):
        return self.tokenizer(text)["input_ids"]

    def test_every_kept_code_is_its_own_token(self):
        codes = sorted(prot_t5.SUPPORTED_RESIDUE_CODES)
        ids = self.ids(" ".join(codes))
        self.assertEqual(ids[-1], self.tokenizer.eos_token_id)
        self.assertNotIn(self.tokenizer.unk_token_id, ids)
        self.assertEqual(len(set(ids[:-1])), len(codes))

    def test_codes_mapped_to_x_are_untrained_or_unknown(self):
        # B/Z/U/O are in the vocabulary, but the published preprocessing maps
        # them to X; J and alignment punctuation have no token at all.
        unknown = self.tokenizer.unk_token_id
        for code in "BZUO":
            with self.subTest(code=code):
                self.assertNotEqual(self.ids(code)[0], unknown)
        for code in "J.-":
            with self.subTest(code=code):
                self.assertEqual(self.ids(code)[0], unknown)

    def test_a_cleaned_sequence_gives_one_token_per_residue(self):
        sequence = prot_t5._clean_sequence("MKTAYIAKQRJBZUO")
        ids = self.ids(" ".join(sequence))
        self.assertEqual(len(ids), len(sequence) + 1)
        self.assertNotIn(self.tokenizer.unk_token_id, ids)


if __name__ == "__main__":
    unittest.main()
