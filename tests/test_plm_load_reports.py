"""Transformers weight-load reports: which are hidden, and ESM-2's unused pooler."""
import contextlib
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "src" / "utilities", ROOT / "src" / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Generate_Embeddings  # noqa: E402  (installs the filter on import)

LOGGER = "transformers.modeling_utils"


def report(*entries):
    """A load report in the shape transformers.utils.loading_report logs."""
    lines = ["EsmModel LOAD REPORT from: checkpoint"]
    lines += [f"- {status}:\t{key}" for status, key in entries]
    return "\n".join(lines)


def shown(message):
    record = logging.LogRecord(LOGGER, logging.WARNING, __file__, 1, message, None, None)
    return Generate_Embeddings._CleanLoadReportFilter().filter(record)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def captured_reports():
    handler = _Capture()
    logger = logging.getLogger(LOGGER)
    logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)


class LoadReportFilterTests(unittest.TestCase):
    def test_only_reports_with_nothing_but_unexpected_entries_are_hidden(self):
        self.assertFalse(shown(report(("UNEXPECTED", "lm_head.weight"))))
        # Transformers raises for MISMATCH and CONVERSION right after logging
        # the report, and its error says to read the report above.
        for status in ("MISSING", "MISMATCH", "CONVERSION"):
            with self.subTest(status=status):
                self.assertTrue(shown(report((status, "encoder.weight"))))
                self.assertTrue(shown(report(("UNEXPECTED", "lm_head.weight"),
                                             (status, "encoder.weight"))))

    def test_other_warnings_are_untouched(self):
        self.assertTrue(shown("An unrelated Transformers warning"))

    def test_filter_is_installed_on_the_modeling_utils_logger(self):
        self.assertTrue(any(
            isinstance(item, Generate_Embeddings._CleanLoadReportFilter)
            for item in logging.getLogger(LOGGER).filters
        ))


class Esm2PoolerTests(unittest.TestCase):
    def test_esm2_loads_without_a_pooler_or_a_missing_report(self):
        import torch
        import transformers
        from transformers import EsmConfig, EsmForMaskedLM
        from resources.pLM_models import esm2

        config = EsmConfig(
            vocab_size=33, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, max_position_embeddings=64, pad_token_id=1,
            mask_token_id=32, position_embedding_type="rotary",
        )
        with tempfile.TemporaryDirectory() as temp:
            # Shaped like facebook/esm2_*: esm.* and lm_head.* weights, no pooler.
            torch.manual_seed(0)
            EsmForMaskedLM(config).save_pretrained(temp)

            # Control: the previous call builds the pooler and its MISSING
            # report reaches the logger, so the check below can see one.
            with captured_reports() as control:
                transformers.AutoModel.from_pretrained(temp, dtype=torch.float32)
            self.assertTrue(any("MISSING" in message for message in control))

            with captured_reports() as messages, mock.patch.object(
                transformers.AutoTokenizer, "from_pretrained", return_value="tokenizer"
            ):
                tokenizer, model = esm2.load_model(temp, "cpu")

        self.assertEqual(tokenizer, "tokenizer")
        self.assertIsNone(model.pooler)
        self.assertEqual(next(model.parameters()).dtype, torch.float32)
        self.assertEqual([message for message in messages if "MISSING" in message], [])


if __name__ == "__main__":
    unittest.main()
