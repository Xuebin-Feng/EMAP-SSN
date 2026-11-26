# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Model_Plugins: static plugin metadata, declared execution modes, the
model license gate (always against a temporary acceptance file) and the
switch that keeps a .bin-only Hub checkpoint from fetching a converted copy."""

import contextlib
from datetime import datetime, timezone
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from transformers import BatchEncoding, T5Config, T5EncoderModel


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resources.pLM_models import ankh  # noqa: E402
from tools.tool_helpers import Model_Plugins as PLM_Plugin_Utils  # noqa: E402


class PluginContractTests(unittest.TestCase):
    def test_every_installed_plugin_has_complete_static_declaration(self):
        modes = PLM_Plugin_Utils.discover_model_execution_modes(
            SRC / "resources" / "pLM_models"
        )
        self.assertEqual(modes["esmc_6b"], "remote_api")
        self.assertEqual(modes["esmc_300m"], "local")
        self.assertIn("esm2_t33_650m", modes)

    def test_missing_and_unknown_modes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = pathlib.Path(temp_dir) / "missing.py"
            missing.write_text(
                'SUPPORTED_MODELS = ["local_model"]\n'
                'MODEL_EXECUTION_MODES = {}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                PLM_Plugin_Utils.read_plugin_metadata(missing)

            unknown = pathlib.Path(temp_dir) / "unknown.py"
            unknown.write_text(
                'SUPPORTED_MODELS = ["future_model"]\n'
                'MODEL_EXECUTION_MODES = {"future_model": "cloud"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unknown"):
                PLM_Plugin_Utils.read_plugin_metadata(unknown)


TERMS = {
    "source_url": "https://huggingface.co/ElnaggarLab/ankh-base",
    "license_id": "CC-BY-NC-SA-4.0",
    "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en",
    "restriction": "Non-commercial use only.",
    "requires_acknowledgement": True,
}


class ModelLicenseGateTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.folder = pathlib.Path(temp_dir.name)
        self.path = self.folder / "license.json"

    def accepted(self, model_name="ankh_base", terms=TERMS):
        return PLM_Plugin_Utils.is_model_license_accepted(model_name, terms, path=self.path)

    def test_unaccepted_terms_fail_closed_with_the_accept_command(self):
        with self.assertRaises(PLM_Plugin_Utils.ModelLicenseAcceptanceRequired) as raised:
            PLM_Plugin_Utils.require_model_license_acceptance(
                "ankh_base", TERMS, path=self.path
            )
        self.assertIsInstance(raised.exception, PermissionError)
        self.assertEqual(
            str(raised.exception),
            "Model: ankh_base\n"
            "Weights license: CC-BY-NC-SA-4.0\n"
            "Restriction: Non-commercial use only.\n"
            "Model source: https://huggingface.co/ElnaggarLab/ankh-base\n"
            "License information: "
            "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en\n\n"
            "The EMAP-SSN integration code is Apache-2.0, but the separately "
            "downloaded model weights are not.\n\n"
            "No model files were accessed. To review and accept these terms "
            "from a terminal, run:\n"
            "  python src/tools/Generate_Embeddings.py --accept-model-license ankh_base",
        )
        self.assertFalse(self.path.exists())

    def test_acceptance_holds_only_for_the_same_model_and_terms(self):
        self.path.write_text(
            json.dumps({"other_model": {"terms_fingerprint": "x"}}), encoding="utf-8"
        )
        self.assertFalse(self.accepted())

        PLM_Plugin_Utils.record_model_license_acceptance("ankh_base", TERMS, path=self.path)

        self.assertTrue(self.accepted())
        PLM_Plugin_Utils.require_model_license_acceptance("ankh_base", TERMS, path=self.path)
        store = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(store["other_model"], {"terms_fingerprint": "x"})
        record = store["ankh_base"]
        accepted_at = datetime.fromisoformat(record.pop("accepted_at"))
        self.assertEqual(accepted_at.tzinfo, timezone.utc)
        self.assertEqual(
            record,
            {
                "license_id": TERMS["license_id"],
                "source_url": TERMS["source_url"],
                "license_url": TERMS["license_url"],
                "terms_fingerprint": PLM_Plugin_Utils.model_terms_fingerprint(
                    "ankh_base", TERMS
                ),
            },
        )
        self.assertEqual(sorted(item.name for item in self.folder.iterdir()), ["license.json"])

        changed = dict(TERMS, license_url="https://example.org/new-license")
        self.assertFalse(self.accepted(terms=changed))
        with self.assertRaises(PLM_Plugin_Utils.ModelLicenseAcceptanceRequired):
            PLM_Plugin_Utils.require_model_license_acceptance(
                "ankh_base", changed, path=self.path
            )
        self.assertFalse(self.accepted(model_name="ankh_large"))

        # The fingerprint binds a record to its model, even copied to another key.
        copied = json.loads(self.path.read_text(encoding="utf-8"))
        copied["ankh_large"] = copied["ankh_base"]
        self.path.write_text(json.dumps(copied), encoding="utf-8")
        self.assertFalse(self.accepted(model_name="ankh_large"))
        self.assertTrue(self.accepted())

    def test_terms_without_acknowledgement_need_no_record(self):
        terms = dict(TERMS, requires_acknowledgement=False)
        self.assertTrue(self.accepted(terms=terms))
        self.assertTrue(self.accepted(terms=None))
        PLM_Plugin_Utils.record_model_license_acceptance("ankh_base", terms, path=self.path)
        self.assertFalse(self.path.exists())

    def test_unreadable_store_counts_as_not_accepted(self):
        for text in ("{", "[]", '{"ankh_base": "accepted"}'):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                self.assertFalse(self.accepted())

    def test_usage_terms_validation_errors(self):
        fields = (
            "['license_id', 'license_url', 'requires_acknowledgement', "
            "'restriction', 'source_url']"
        )
        cases = {
            "not a mapping": (
                [TERMS],
                "MODEL_USAGE_TERMS must be a literal mapping.",
            ),
            "unsupported model": (
                {"ankh_large": TERMS},
                "MODEL_USAGE_TERMS contains unsupported model(s): ['ankh_large'].",
            ),
            "missing field": (
                {"ankh_base": {key: value for key, value in TERMS.items() if key != "restriction"}},
                f"MODEL_USAGE_TERMS['ankh_base'] must contain exactly {fields}.",
            ),
            "extra field": (
                {"ankh_base": dict(TERMS, version="1")},
                f"MODEL_USAGE_TERMS['ankh_base'] must contain exactly {fields}.",
            ),
            "blank string": (
                {"ankh_base": dict(TERMS, license_id="  ")},
                "MODEL_USAGE_TERMS['ankh_base']['license_id'] must be a non-empty string.",
            ),
            "non-string": (
                {"ankh_base": dict(TERMS, license_url=5)},
                "MODEL_USAGE_TERMS['ankh_base']['license_url'] must be a non-empty string.",
            ),
            "non-boolean flag": (
                {"ankh_base": dict(TERMS, requires_acknowledgement="yes")},
                "MODEL_USAGE_TERMS['ankh_base'] acknowledgement flag must be boolean.",
            ),
        }
        for label, (usage_terms, message) in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError) as raised:
                    PLM_Plugin_Utils.validate_model_usage_terms(["ankh_base"], usage_terms)
                self.assertEqual(str(raised.exception), message)

        valid = {"ankh_base": TERMS}
        self.assertIs(PLM_Plugin_Utils.validate_model_usage_terms(["ankh_base"], valid), valid)
        self.assertEqual(PLM_Plugin_Utils.validate_model_usage_terms(["ankh_base"], None), {})


class ModelDisplayNameTests(unittest.TestCase):
    """A plugin names its models for the Tools dropdowns; identifiers stay what files hold."""

    SHIPPED = {
        "ankh_base": "Ankh Base",
        "ankh_large": "Ankh Large",
        "ankh3_large": "Ankh3 Large",
        "ankh3_large_s2s": "Ankh3 Large (S2S)",
        "ankh3_xl": "Ankh3 XL",
        "ankh3_xl_s2s": "Ankh3 XL (S2S)",
        "esm2_t6_8m": "ESM-2 8M",
        "esm2_t12_35m": "ESM-2 35M",
        "esm2_t30_150m": "ESM-2 150M",
        "esm2_t33_650m": "ESM-2 650M",
        "esmc_300m": "ESM C 300M",
        "esmc_600m": "ESM C 600M",
        "esmc_6b": "ESM C 6B",
        "prost_t5": "ProstT5",
        "prot_bert": "ProtBERT",
        "prot_t5": "ProtT5",
    }

    def test_every_shipped_model_has_the_name_the_dropdowns_show(self):
        plugin_dir = SRC / "resources" / "pLM_models"
        names = PLM_Plugin_Utils.discover_model_display_names(plugin_dir)
        self.assertEqual(names, self.SHIPPED)
        # Every model the plugins support is named, and none is named by its identifier.
        modes = PLM_Plugin_Utils.discover_model_execution_modes(plugin_dir)
        self.assertEqual(set(names), set(modes))
        for model, name in names.items():
            with self.subTest(model=model):
                self.assertNotEqual(name, model)
                self.assertNotIn("_", name)

    def test_a_plugin_that_declares_no_names_shows_its_identifiers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin = pathlib.Path(temp_dir) / "plain.py"
            plugin.write_text(
                'SUPPORTED_MODELS = ["plain_model"]\n'
                'MODEL_EXECUTION_MODES = {"plain_model": "local"}\n',
                encoding="utf-8",
            )
            self.assertEqual(PLM_Plugin_Utils.read_model_display_names(plugin), {})
            self.assertEqual(PLM_Plugin_Utils.discover_model_display_names(temp_dir), {})
        self.assertEqual(PLM_Plugin_Utils.format_model_selector_label("plain_model", None), "plain_model")

    def test_declared_names_are_read_without_importing_the_plugin(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin = pathlib.Path(temp_dir) / "named.py"
            plugin.write_text(
                'import this_module_does_not_exist\n'
                'SUPPORTED_MODELS = ["m_a", "m_b"]\n'
                'MODEL_EXECUTION_MODES = {"m_a": "local", "m_b": "local"}\n'
                'MODEL_DISPLAY_NAMES = {"m_a": "Model A"}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                PLM_Plugin_Utils.discover_model_display_names(temp_dir), {"m_a": "Model A"}
            )

    def test_validation_errors(self):
        cases = {
            "not a mapping": (["Model"], "MODEL_DISPLAY_NAMES must be a literal mapping."),
            "unknown model": (
                {"other": "Other"},
                "MODEL_DISPLAY_NAMES contains unsupported model(s): ['other'].",
            ),
            "blank": (
                {"model": "  "},
                "MODEL_DISPLAY_NAMES['model'] must be a non-empty string without leading or trailing spaces.",
            ),
            "padded": (
                {"model": " Model"},
                "MODEL_DISPLAY_NAMES['model'] must be a non-empty string without leading or trailing spaces.",
            ),
            "non-string": (
                {"model": 5},
                "MODEL_DISPLAY_NAMES['model'] must be a non-empty string without leading or trailing spaces.",
            ),
        }
        for label, (names, message) in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError) as raised:
                    PLM_Plugin_Utils.validate_model_display_names(["model"], names)
                self.assertEqual(str(raised.exception), message)
        with self.assertRaisesRegex(ValueError, "same name"):
            PLM_Plugin_Utils.validate_model_display_names(
                ["a", "b"], {"a": "Same", "b": "Same"}
            )
        valid = {"model": "Model"}
        self.assertIs(PLM_Plugin_Utils.validate_model_display_names(["model"], valid), valid)
        self.assertEqual(PLM_Plugin_Utils.validate_model_display_names(["model"], None), {})

    def test_two_plugins_cannot_share_a_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for index in (1, 2):
                pathlib.Path(temp_dir, f"p{index}.py").write_text(
                    f'SUPPORTED_MODELS = ["model_{index}"]\n'
                    f'MODEL_EXECUTION_MODES = {{"model_{index}": "local"}}\n'
                    'MODEL_DISPLAY_NAMES = {"model_%d": "Same Name"}\n' % index,
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "same display name"):
                PLM_Plugin_Utils.discover_model_display_names(temp_dir)

    def test_a_loaded_plugin_with_a_bad_name_table_is_rejected(self):
        plugin = type("Plugin", (), {
            "SUPPORTED_MODELS": ["model"],
            "MODEL_EXECUTION_MODES": {"model": "local"},
            "MODEL_DISPLAY_NAMES": {"model": ""},
        })
        with self.assertRaisesRegex(ValueError, "MODEL_DISPLAY_NAMES"):
            PLM_Plugin_Utils.validate_loaded_plugin(plugin, "model")
        plugin.MODEL_DISPLAY_NAMES = {"model": "Model"}
        self.assertEqual(PLM_Plugin_Utils.validate_loaded_plugin(plugin, "model"), "local")

    def test_the_selector_label_names_the_model_and_keeps_the_terms_marker(self):
        label = PLM_Plugin_Utils.format_model_selector_label
        # Without separate terms the label is the name alone, a plain string.
        self.assertEqual(label("esm2_t6_8m", None, "ESM-2 8M"), "ESM-2 8M")
        self.assertIsInstance(label("esm2_t6_8m", None, "ESM-2 8M"), str)
        non_commercial = label("ankh_base", dict(TERMS), "Ankh Base")
        self.assertEqual(str(non_commercial), "Ankh Base [non-commercial]")
        self.assertEqual(non_commercial.template, "{model} [non-commercial]")
        other = label("some_model", dict(TERMS, restriction="Custom terms apply."), "Some Model")
        self.assertEqual(str(other), "Some Model [separate terms]")
        # No name: the identifier, as before.
        self.assertEqual(str(label("ankh_base", dict(TERMS))), "ankh_base [non-commercial]")

    def test_the_shipped_plugins_still_load_by_identifier(self):
        for model in self.SHIPPED:
            with self.subTest(model=model):
                plugin = PLM_Plugin_Utils.find_model_plugin(model)
                self.assertIn(model, plugin.SUPPORTED_MODELS)
                self.assertEqual(plugin.MODEL_DISPLAY_NAMES[model], self.SHIPPED[model])


class SafetensorsConversionSwitchTests(unittest.TestCase):
    """A .bin-only Hub checkpoint loads without fetching the converter bot's copy."""

    SWITCH = "DISABLE_SAFETENSORS_CONVERSION"

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_switch_is_on_inside_and_removed_after(self):
        os.environ.pop(self.SWITCH, None)
        with PLM_Plugin_Utils.safetensors_conversion_disabled():
            self.assertEqual(os.environ[self.SWITCH], "1")
        self.assertNotIn(self.SWITCH, os.environ)

    def test_a_value_set_before_comes_back_even_after_an_error(self):
        os.environ[self.SWITCH] = "0"
        with self.assertRaises(RuntimeError):
            with PLM_Plugin_Utils.safetensors_conversion_disabled():
                raise RuntimeError("load failed")
        self.assertEqual(os.environ[self.SWITCH], "0")

    def test_transformers_still_reads_this_switch(self):
        # Transformers decides whether to start the conversion thread from this
        # variable; if an upgrade renames it, the helper silently stops working.
        import inspect
        from transformers import modeling_utils
        from transformers.utils.import_utils import is_env_variable_true

        self.assertEqual(PLM_Plugin_Utils.SAFETENSORS_CONVERSION_SWITCH, self.SWITCH)
        self.assertIn(f'is_env_variable_true("{self.SWITCH}")', inspect.getsource(modeling_utils))
        with PLM_Plugin_Utils.safetensors_conversion_disabled():
            self.assertTrue(is_env_variable_true(self.SWITCH))

    def test_every_bin_only_rostlab_checkpoint_loads_inside_the_switch(self):
        # Rostlab/prot_bert, Rostlab/ProstT5 and Rostlab/prot_t5_xl_uniref50 hold
        # only pytorch_model.bin; without the switch their first load also
        # fetches the bot's safetensors copy and the tool cannot exit until it ends.
        import importlib

        for name, tokenizer, model in (("prot_bert", "BertTokenizer", "BertModel"),
                                       ("prost_t5", "T5Tokenizer", "T5EncoderModel"),
                                       ("prot_t5", "T5Tokenizer", "T5EncoderModel")):
            with self.subTest(plugin=name):
                plugin = importlib.import_module(f"resources.pLM_models.{name}")
                seen = {}

                def load(*args, **kwargs):
                    seen["during"] = os.environ.get(self.SWITCH)
                    return mock.Mock()

                os.environ.pop(self.SWITCH, None)
                with mock.patch(f"transformers.{tokenizer}.from_pretrained", return_value="tokenizer"), \
                        mock.patch(f"transformers.{model}.from_pretrained", side_effect=load) as loader:
                    plugin.load_model(name, "cpu")
                loader.assert_called_once()
                self.assertEqual(seen["during"], "1")
                self.assertNotIn(self.SWITCH, os.environ)


class _FakeAnkhTokenizer:
    """In-memory tokenizer mimicking Ankh's unspaced bare-AA vocab and T5 Metaspace behavior."""

    eos_token_id = 1
    unk_token_id = 2

    def __init__(self, *, pre_tokenizer=None, prepend_unk=False):
        self.vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
        for offset, ch in enumerate("ACDEFGHIKLMNPQRSTVWYXBZUO", start=3):
            self.vocab[ch] = offset
        self.prepend_unk = prepend_unk
        self.seen_inputs = []

        class _Backend:
            def __init__(self, initial_pre_tokenizer):
                self.pre_tokenizer = initial_pre_tokenizer

        self._tokenizer = _Backend(pre_tokenizer)

    def _encode_one(self, text):
        pre_tok = self._tokenizer.pre_tokenizer
        if pre_tok is not None:
            # Run the real tokenizers pre-tokenizer so prepend_scheme="always"
            # inserts "▁" (absent from Ankh's vocab -> <unk>) and "never" does not.
            pieces = [piece for piece, _ in pre_tok.pre_tokenize_str(text)]
            ids = [self.vocab.get(ch, self.unk_token_id) for piece in pieces for ch in piece]
        else:
            ids = [self.unk_token_id] if self.prepend_unk else []
            for ch in text:
                ids.append(self.vocab.get(ch, self.unk_token_id))
        ids.append(self.eos_token_id)
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


class AnkhPluginTests(unittest.TestCase):
    """Direct unit tests for the Ankh v1 pLM plugin (src/resources/pLM_models/ankh.py)."""

    def _tiny_encoder(self):
        torch.manual_seed(0)
        cfg = T5Config(
            vocab_size=32,
            d_model=32,
            d_kv=8,
            d_ff=64,
            num_layers=2,
            num_heads=4,
            pad_token_id=0,
            eos_token_id=1,
            feed_forward_proj="gated-gelu",
        )
        return T5EncoderModel(cfg).to(dtype=torch.float32).eval()

    def test_clean_sequence_handles_standard_ambiguous_nonstandard_and_trimming(self):
        self.assertEqual(
            ankh._clean_sequence("acdefghiklmnpqrstvwyxbzuo"),
            "ACDEFGHIKLMNPQRSTVWYXBZUO",
        )
        self.assertEqual(ankh._clean_sequence("--??..**mkta**..??--"), "MKTA")
        self.assertEqual(ankh._clean_sequence("  --a--  "), "A")
        self.assertEqual(ankh._clean_sequence("**a-b.j*xuzoc**"), "AXBXXXXUZOC")
        self.assertEqual(ankh._clean_sequence("m k\tt-a"), "MXKXTXA")
        self.assertEqual(ankh._clean_sequence("---...***   "), "")
        self.assertEqual(ankh._clean_sequence(""), "")

    def test_tokenized_residue_count_and_already_aligned_tokenizer_untouched(self):
        aligned_tok = _FakeAnkhTokenizer(pre_tokenizer=None, prepend_unk=False)
        self.assertEqual(ankh._tokenized_residue_count(aligned_tok), len(ankh._ALIGNMENT_PROBE))
        result = ankh._residue_aligned_tokenizer(aligned_tok)
        self.assertIs(result, aligned_tok)
        self.assertIsNone(aligned_tok._tokenizer.pre_tokenizer)

    def test_residue_aligned_tokenizer_repairs_transformers5_metaspace_leading_unk(self):
        from tokenizers.pre_tokenizers import Metaspace

        bad_pre = Metaspace(replacement="▁", prepend_scheme="always", split=True)
        tok = _FakeAnkhTokenizer(pre_tokenizer=bad_pre)
        before_ids = tok(ankh._ALIGNMENT_PROBE)["input_ids"]
        self.assertEqual(before_ids[0], tok.unk_token_id)
        self.assertEqual(ankh._tokenized_residue_count(tok), len(ankh._ALIGNMENT_PROBE) + 1)

        repaired = ankh._residue_aligned_tokenizer(tok)
        self.assertIs(repaired, tok)
        self.assertEqual(ankh._tokenized_residue_count(tok), len(ankh._ALIGNMENT_PROBE))

        state = json.loads(tok._tokenizer.pre_tokenizer.__getstate__().decode("utf-8"))
        self.assertEqual(state["type"], "Sequence")
        metaspace_cfg = state["pretokenizers"][1]
        self.assertEqual(metaspace_cfg["type"], "Metaspace")
        self.assertEqual(metaspace_cfg["prepend_scheme"], "never")

        after_ids = tok(ankh._ALIGNMENT_PROBE)["input_ids"]
        self.assertEqual(len(after_ids), len(ankh._ALIGNMENT_PROBE) + 1)
        self.assertNotIn(tok.unk_token_id, after_ids)
        self.assertEqual(after_ids[-1], tok.eos_token_id)

    def test_residue_aligned_tokenizer_raises_when_unrepairable_or_still_misaligned(self):
        class _NoBackendTokenizer:
            def __call__(self, text):
                return {"input_ids": [2, 10, 11, 12, 13, 1]}

        with self.subTest("cannot remove prefix"), self.assertRaisesRegex(
            RuntimeError, "word-start prefix could not be removed"
        ):
            ankh._residue_aligned_tokenizer(_NoBackendTokenizer())

        class _StillMisalignedTokenizer:
            def __init__(self):
                self._tokenizer = type("_Backend", (), {"pre_tokenizer": None})()

            def __call__(self, text):
                return {"input_ids": [2, 10, 11, 12, 13, 1]}

        with self.subTest("still fails probe after repair"), self.assertRaisesRegex(
            RuntimeError, "not residue-aligned"
        ):
            ankh._residue_aligned_tokenizer(_StillMisalignedTokenizer())

    def test_load_model_enforces_license_gate_before_touching_hub(self):
        for model_name in ankh.SUPPORTED_MODELS:
            with self.subTest(model_name=model_name):
                with mock.patch.object(
                    ankh,
                    "require_model_license_acceptance",
                    side_effect=PLM_Plugin_Utils.ModelLicenseAcceptanceRequired("accept first"),
                ) as gate, mock.patch(
                    "transformers.AutoTokenizer.from_pretrained"
                ) as tok_loader, mock.patch(
                    "transformers.T5EncoderModel.from_pretrained"
                ) as model_loader:
                    with self.assertRaises(PLM_Plugin_Utils.ModelLicenseAcceptanceRequired):
                        ankh.load_model(model_name, torch.device("cpu"))
                gate.assert_called_once_with(model_name, ankh.MODEL_USAGE_TERMS[model_name])
                tok_loader.assert_not_called()
                model_loader.assert_not_called()

    def test_load_model_loads_auto_tokenizer_and_fp32_t5_encoder_on_device_in_eval_mode(self):
        from tokenizers.pre_tokenizers import Metaspace

        expected_repos = {
            "ankh_base": "ElnaggarLab/ankh-base",
            "ankh_large": "ElnaggarLab/ankh-large",
        }
        device = torch.device("cpu")
        for model_name, repo in expected_repos.items():
            with self.subTest(model_name=model_name):
                bad_pre = Metaspace(replacement="▁", prepend_scheme="always", split=True)
                fake_tok = _FakeAnkhTokenizer(pre_tokenizer=bad_pre)
                fake_model = mock.Mock()
                fake_model.to.return_value = fake_model

                with mock.patch.object(
                    ankh, "require_model_license_acceptance"
                ) as gate, mock.patch(
                    "transformers.AutoTokenizer.from_pretrained", return_value=fake_tok
                ) as tok_loader, mock.patch(
                    "transformers.T5EncoderModel.from_pretrained", return_value=fake_model
                ) as model_loader, contextlib.redirect_stdout(io.StringIO()):
                    tok, mod = ankh.load_model(model_name, device)

                gate.assert_called_once_with(model_name, ankh.MODEL_USAGE_TERMS[model_name])
                tok_loader.assert_called_once_with(repo)
                model_loader.assert_called_once_with(repo, dtype=torch.float32)
                fake_model.to.assert_called_once_with(device)
                fake_model.eval.assert_called_once_with()
                self.assertIs(tok, fake_tok)
                self.assertIs(mod, fake_model)
                self.assertEqual(ankh._tokenized_residue_count(tok), len(ankh._ALIGNMENT_PROBE))

    def test_get_embedding_and_batch_slice_residues_in_fp32_and_convert_target_dtype(self):
        model = self._tiny_encoder()
        tok = _FakeAnkhTokenizer(pre_tokenizer=None, prepend_unk=False)
        device = torch.device("cpu")
        model_obj = (tok, model)
        seqs = ["ACDEFGHIKLMNPQRSTVWY", "MKTA-BZJ"]

        # Exact single-vs-batch equivalence and residue lengths (20 and 8).
        single = [ankh.get_embedding(s, model_obj, device, np.float32) for s in seqs]
        batched = ankh.get_embeddings_batch(seqs, model_obj, device, np.float32)
        self.assertEqual([a.shape for a in single], [(20, 32), (8, 32)])
        for a, b in zip(single, batched):
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, np.dtype(np.float32))
            self.assertEqual(b.dtype, np.dtype(np.float32))
            self.assertLess(float(np.max(np.abs(a - b))), 1e-5)

        # Spy on model.forward to verify strict FP32 compute and exact :-1 / 0:length slicing
        # (no trailing </s> and no extra prefix token) even when storing float16.
        captured = []
        orig_forward = model.forward

        def spy_forward(**kwargs):
            out = orig_forward(**kwargs)
            captured.append(out.last_hidden_state.detach().clone())
            return out

        with mock.patch.object(model, "forward", side_effect=spy_forward):
            single_f32 = ankh.get_embedding("MKTA", model_obj, device, np.float32)
            single_f16 = ankh.get_embedding("MKTA", model_obj, device, np.float16)
            batch_f16 = ankh.get_embeddings_batch(["MKTA", "ACDEF"], model_obj, device, np.float16)

        self.assertEqual(len(captured), 3)
        for hidden in captured:
            self.assertEqual(hidden.dtype, torch.float32)

        # Single forward received 4 residues + 1 trailing </s> = 5 tokens (no prefix token).
        self.assertEqual(tuple(captured[0].shape), (1, 5, 32))
        np.testing.assert_allclose(single_f32, captured[0][0, :4].numpy(), atol=1e-6)
        self.assertEqual((single_f16.shape, single_f16.dtype), ((4, 32), np.dtype(np.float16)))
        np.testing.assert_array_equal(single_f16, captured[1][0, :4].to(torch.float16).numpy())

        # Batch forward received max(4, 5) + 1 trailing </s> = 6 tokens; each row sliced at offset 0.
        self.assertEqual(tuple(captured[2].shape), (2, 6, 32))
        self.assertEqual([(a.shape, a.dtype) for a in batch_f16], [
            ((4, 32), np.dtype(np.float16)),
            ((5, 32), np.dtype(np.float16)),
        ])
        np.testing.assert_array_equal(batch_f16[0], captured[2][0, :4].to(torch.float16).numpy())
        np.testing.assert_array_equal(batch_f16[1], captured[2][1, :5].to(torch.float16).numpy())

    def test_empty_sequences_handling_and_value_error(self):
        model = self._tiny_encoder()
        tok = _FakeAnkhTokenizer(pre_tokenizer=None, prepend_unk=False)
        model_obj = (tok, model)
        device = torch.device("cpu")

        self.assertEqual(ankh.get_embeddings_batch([], model_obj, device, np.float32), [])
        with self.assertRaisesRegex(ValueError, "no supported amino-acid"):
            ankh.get_embedding("---***", model_obj, device, np.float32)
        with self.assertRaisesRegex(ValueError, "no supported amino-acid"):
            ankh.get_embeddings_batch(["MKTA", "---***"], model_obj, device, np.float32)
        self.assertEqual(tok.seen_inputs, [])


if __name__ == "__main__":
    unittest.main()
