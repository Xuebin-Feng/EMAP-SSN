"""Tests of Embedding_Injection: model-name to embedding-plugin resolution and
the load_model/get_embedding delegation to the selected plugin.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
UTILITIES_DIR = os.path.join(PROJECT_ROOT, "src", "utilities")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "src", "tools")
for path in (SRC_DIR, UTILITIES_DIR, TOOLS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# The tests package points tool imports at a missing settings file.
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Embedding_Injection as embedding_injection


class EmbeddingInjectionPluginTests(unittest.TestCase):
    EXPECTED_PLUGINS = {
        "ankh_base": "ankh",
        "ankh_large": "ankh",
        "esm2_t6_8m": "esm2",
        "esm2_t12_35m": "esm2",
        "esm2_t30_150m": "esm2",
        "esm2_t33_650m": "esm2",
        "esmc_300m": "esmc",
        "esmc_600m": "esmc",
        "esmc_6b": "esmc_6b_api",
        "prost_t5": "prost_t5",
        "prot_bert": "prot_bert",
    }

    def test_every_declared_model_resolves_to_the_expected_plugin(self):
        for model_name, expected_module in self.EXPECTED_PLUGINS.items():
            with self.subTest(model_name=model_name):
                plugin = embedding_injection.find_model_plugin(model_name)

                self.assertIsNotNone(plugin)
                self.assertEqual(plugin.__name__, expected_module)
                self.assertTrue(callable(plugin.load_model))
                self.assertTrue(callable(plugin.get_embedding))
                if model_name == "esmc_6b":
                    # The 6B model is served by the remote API plugin.
                    self.assertIn("API_MODEL_MAPPINGS", vars(plugin))

    def test_load_model_delegates_to_selected_plugin(self):
        plugin = mock.Mock()
        plugin.__name__ = "test_plugin"
        plugin.load_model.return_value = mock.sentinel.model
        plugin.get_embedding = mock.Mock()

        with mock.patch.object(
            embedding_injection,
            "find_model_plugin",
            return_value=plugin,
        ), mock.patch.object(
            embedding_injection.Hardware_Utils,
            "get_optimal_device",
            return_value=mock.sentinel.device,
        ):
            model_obj, device, selected_plugin = embedding_injection.load_model(
                "test_model"
            )

        plugin.load_model.assert_called_once_with(
            "test_model",
            mock.sentinel.device,
        )
        self.assertIs(model_obj, mock.sentinel.model)
        self.assertIs(device, mock.sentinel.device)
        self.assertIs(selected_plugin, plugin)

    def test_get_embedding_delegates_sequence_processing_to_plugin(self):
        expected = np.ones((3, 5), dtype=np.float16)
        plugin = mock.Mock()
        plugin.get_embedding.return_value = expected

        actual = embedding_injection.get_embedding(
            "AB-CD",
            mock.sentinel.model,
            mock.sentinel.device,
            plugin,
            np.float16,
        )

        plugin.get_embedding.assert_called_once_with(
            "AB-CD",
            mock.sentinel.model,
            mock.sentinel.device,
            np.float16,
        )
        self.assertIs(actual, expected)

    def test_unknown_model_reports_unsupported_plugin(self):
        with mock.patch.object(
            embedding_injection,
            "find_model_plugin",
            return_value=None,
        ), self.assertRaisesRegex(
            ValueError,
            "not supported by any available plugin",
        ):
            embedding_injection.load_model("not_a_model")


if __name__ == "__main__":
    unittest.main()
