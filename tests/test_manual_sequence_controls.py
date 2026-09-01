"""Backend tests for explicit manual-sequence switches."""

import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import h5py
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tools.Embedding_PWA import (
    prepare_embedding_database,
    resolve_manual_alignment_inputs,
    run_alignment,
    sanitize_alignment_header,
)
import tools.Embedding_PWA as embedding_pwa
from tools.Embedding_SSEARCH import resolve_manual_query_sequence


class ManualSequenceControlTests(unittest.TestCase):
    def test_pairwise_has_no_fasta_configuration(self):
        self.assertFalse(hasattr(embedding_pwa, "INPUT_FASTA"))
        self.assertFalse(hasattr(embedding_pwa, "FULL_INPUT_FASTA"))
        self.assertFalse(hasattr(embedding_pwa, "FASTA_DIR"))

    def test_pairwise_ignores_saved_text_while_switches_are_off(self):
        ref, tar, model = resolve_manual_alignment_inputs(
            False,
            "SAVED_REF",
            False,
            "SAVED_TAR",
            "database_model",
            "manual_model",
        )
        self.assertEqual((ref, tar, model), ("", "", "database_model"))

    def test_pairwise_uses_database_model_if_only_one_sequence_is_manual(self):
        ref, tar, model = resolve_manual_alignment_inputs(
            True,
            "REF",
            False,
            "SAVED_TAR",
            "database_model",
            "manual_model",
        )
        self.assertEqual((ref, tar, model), ("REF", "", "database_model"))

    def test_pairwise_uses_selected_model_if_both_sequences_are_manual(self):
        ref, tar, model = resolve_manual_alignment_inputs(
            True,
            "REF",
            True,
            "TAR",
            "database_model",
            "manual_model",
        )
        self.assertEqual((ref, tar, model), ("REF", "TAR", "manual_model"))

    def test_pairwise_manual_inputs_use_canonical_sanitization(self):
        ref, tar, model = resolve_manual_alignment_inputs(
            True,
            "***ac-d?***",
            True,
            " bz u ",
            "database_model",
            "manual_model",
        )
        self.assertEqual((ref, tar, model), ("ACXD", "BZXU", "manual_model"))
        self.assertEqual(
            sanitize_alignment_header(' protein[1]/chain  A '),
            "protein(1)_chain_A",
        )

    def test_pairwise_rejects_enabled_empty_manual_sequence(self):
        with self.assertRaisesRegex(ValueError, "Reference manual sequence"):
            resolve_manual_alignment_inputs(
                True,
                "***",
                True,
                "TARGET",
                "database_model",
                "manual_model",
            )

    def test_pairwise_reads_sanitized_sequences_from_hdf5_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "database.h5")
            headers = ["reference(1)_A", "target"]
            sequences = ["ACDX", "ACDE"]
            with h5py.File(database_path, "w") as hf:
                hf.attrs["model_name"] = "test_model"
                hf.attrs["saving_mode"] = "float32"
                hf.attrs["num_sequences"] = 2
                hf.attrs["generation_complete"] = True
                hf.create_dataset("headers", data=np.asarray(headers, dtype="S"))
                hf.create_dataset("sequences", data=np.asarray(sequences, dtype="S"))
                embeddings = hf.create_group("embeddings")
                embeddings.create_dataset(
                    headers[0],
                    data=np.arange(12, dtype=np.float32).reshape(4, 3),
                )
                embeddings.create_dataset(
                    headers[1],
                    data=np.arange(12, 24, dtype=np.float32).reshape(4, 3),
                )

            database = prepare_embedding_database(database_path)
            self.assertEqual(database.model_name, "test_model")
            self.assertEqual(database.sequence_by_header, dict(zip(headers, sequences)))

            with mock.patch.object(embedding_pwa, "GENERATE_REPORT", False):
                with redirect_stdout(StringIO()) as output:
                    run_alignment(
                        " reference[1]/A ",
                        "target",
                        "",
                        "",
                        database_path,
                        database.sequence_by_header,
                        "global",
                        -2.0,
                        0.0,
                        "",
                        database.model_name,
                    )
            self.assertIn("reference(1)_A", output.getvalue())
            self.assertIn("Found pre-calculated embedding", output.getvalue())

    def test_database_search_ignores_saved_text_while_switch_is_off(self):
        self.assertEqual(resolve_manual_query_sequence(False, " SAVED "), "")
        self.assertEqual(resolve_manual_query_sequence(True, " QUERY "), "QUERY")


if __name__ == "__main__":
    unittest.main()
