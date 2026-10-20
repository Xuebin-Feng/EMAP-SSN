# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Model_Plugins: static plugin metadata, declared execution modes and the
model license gate (always against a temporary acceptance file)."""

from datetime import datetime, timezone
import json
import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
        "esm2_t6_8m": "ESM-2 8M",
        "esm2_t12_35m": "ESM-2 35M",
        "esm2_t30_150m": "ESM-2 150M",
        "esm2_t33_650m": "ESM-2 650M",
        "esmc_300m": "ESM C 300M",
        "esmc_600m": "ESM C 600M",
        "esmc_6b": "ESM C 6B",
        "prost_t5": "ProstT5",
        "prot_bert": "ProtBERT",
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


if __name__ == "__main__":
    unittest.main()
