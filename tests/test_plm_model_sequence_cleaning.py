import importlib.util
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "src" / "resources" / "pLM_models"


def load_plugin(module_name):
    module_path = MODEL_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelSequenceCleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugins = {
            name: load_plugin(name)
            for name in (
                "ankh",
                "esm2",
                "esmc",
                "esmc_6b_api",
                "prost_t5",
                "prot_bert",
            )
        }

    def test_esm_models_preserve_native_ambiguity_and_alignment_tokens(self):
        source = "**A.B-JXUZOC**"
        expected = "A.B-XXUZOC"

        for name in ("esm2", "esmc", "esmc_6b_api"):
            with self.subTest(plugin=name):
                self.assertEqual(self.plugins[name]._clean_sequence(source), expected)

    def test_ankh_preserves_native_ambiguity_codes_but_not_gap_or_j(self):
        self.assertEqual(
            self.plugins["ankh"]._clean_sequence("**A-BJXUZOC**"),
            "AXBXXUZOC",
        )

    def test_prottrans_models_apply_their_published_ambiguity_mapping(self):
        source = "**A-BJXUZOC**"
        expected = "AXXXXXXXC"

        for name in ("prost_t5", "prot_bert"):
            with self.subTest(plugin=name):
                self.assertEqual(self.plugins[name]._clean_sequence(source), expected)

    def test_cleaners_uppercase_sequences(self):
        for name, plugin in self.plugins.items():
            with self.subTest(plugin=name):
                self.assertEqual(plugin._clean_sequence("acd"), "ACD")

    def test_cleaners_trim_non_residue_edges(self):
        for name, plugin in self.plugins.items():
            with self.subTest(plugin=name):
                self.assertEqual(plugin._clean_sequence("--??ACD??--"), "ACD")

    def test_cleaners_reject_inputs_without_residue_codes(self):
        for name, plugin in self.plugins.items():
            with self.subTest(plugin=name):
                self.assertEqual(plugin._clean_sequence("---...***"), "")


if __name__ == "__main__":
    unittest.main()
