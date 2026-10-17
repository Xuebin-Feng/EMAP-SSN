# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Configuration window: the Consistency Check report.

The check reads FASTA, network and MSA headers with the canonical sanitization,
reports whether the FASTA is a subset of the network and how much of it the MSA
covers, and resolves the alignment reference as the Viewer does.
"""

import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import h5py


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import EMAPSSN_Config  # noqa: E402
from tests.config_gui_loader import load_config_namespace  # noqa: E402
from tests.embedding_fixtures import write_embedding_file  # noqa: E402

_GUI_CLASS = None
_APP = None


def setUpModule():
    # run_consistency_check needs only the class; no window (and no settings) is opened.
    global _GUI_CLASS, _APP
    namespace = load_config_namespace()
    _APP = namespace["QApplication"].instance() or namespace["QApplication"]([])
    _GUI_CLASS = namespace["ConfigGUI"]


class ConsistencyHeaderSanitizationTests(unittest.TestCase):
    def test_fasta_headers_use_canonical_sanitization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = pathlib.Path(temp_dir) / "sequences.fasta"
            fasta_path.write_text(
                ">  Alpha   Beta  \nac-d\n",
                encoding="utf-8",
            )

            headers = EMAPSSN_Config._load_consistency_fasta_headers(fasta_path)

        self.assertEqual(headers, ["Alpha_Beta"])

    def test_fasta_and_sparse_msa_headers_share_the_canonical_rule(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            msa_fasta_path = temp_path / "alignment.fasta"
            msa_hdf5_path = temp_path / "alignment.h5"
            msa_fasta_path.write_text(
                ">Alpha   Beta\nAC-D\n",
                encoding="utf-8",
            )
            with h5py.File(msa_hdf5_path, "w") as hf:
                hf.create_dataset("headers", data=[b"Alpha   Beta"])

            fasta_headers = EMAPSSN_Config._load_consistency_msa_headers(
                msa_fasta_path
            )
            hdf5_headers = EMAPSSN_Config._load_consistency_msa_headers(
                msa_hdf5_path
            )

        self.assertEqual(fasta_headers, ["Alpha_Beta"])
        self.assertEqual(hdf5_headers, ["Alpha_Beta"])


class _Choice:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text

    def text(self):
        return self._text


class _Panel:
    message = ""

    def setText(self, text):
        self.message = text


class _ReportFixture:
    """A FASTA, network and MSA in a folder, and the report a Consistency Check gives."""

    # Distinct sequences: the canonical FASTA reader drops exact duplicates.
    RECORDS = {
        "WP_0123.1_protein_A": "MKTAYIAK",
        "WP_0123.10_protein_B": "MSEQNNTE",
        "Other_seq": "MAAAGGGK",
    }

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        folder = pathlib.Path(self._temp.name)
        (folder / "nodes.fasta").write_text(
            "".join(f">{header}\n{sequence}\n" for header, sequence in self.RECORDS.items()),
            encoding="utf-8",
        )
        with h5py.File(folder / "network.h5", "w") as hf:
            hf.create_dataset("headers", data=[header.encode() for header in self.RECORDS])
        # The MSA lacks WP_0123.10_protein_B.
        (folder / "alignment.fasta").write_text(
            ">WP_0123.1_protein_A\nMKTAYIAK\n>Other_seq\nMAAAGGGK\n", encoding="utf-8"
        )
        self.folder = str(folder)

    def tearDown(self):
        self._temp.cleanup()

    def report(self, reference, fasta="nodes.fasta", embedding=""):
        panel = _Panel()
        stand_in = SimpleNamespace(
            cb_fasta=_Choice(fasta),
            cb_hdf5=_Choice("network.h5"),
            cb_msa=_Choice("alignment.fasta"),
            cb_embed=_Choice(embedding),
            line_ref=_Choice(reference),
            tip_panel=panel,
            _resolved_directory_input=lambda key: self.folder,
        )
        _GUI_CLASS.run_consistency_check(stand_in)
        self.assertNotIn("Error during consistency check", panel.message)
        return panel.message


class ConsistencyReferenceReportTests(_ReportFixture, unittest.TestCase):
    """The report resolves the reference as the Viewer does, over the network headers."""

    def test_a_leading_identifier_names_one_header_not_every_substring_match(self):
        message = self.report("WP_0123.1")
        self.assertIn(
            "SUCCESS: Reference ID 'WP_0123.1' resolves to WP_0123.1_protein_A.", message
        )
        self.assertNotIn("equally", message)

    def test_tied_matches_name_the_header_used_and_the_others(self):
        message = self.report("WP_01*")
        self.assertIn("resolves to WP_0123.1_protein_A.", message)
        self.assertIn(
            "It matches 2 headers equally; the first in network order is used. "
            "Others: WP_0123.10_protein_B",
            message,
        )

    def test_a_wildcard_is_resolved_rather_than_reported_missing(self):
        message = self.report("*.1")
        self.assertIn("SUCCESS: Reference ID '*.1' resolves to WP_0123.1_protein_A.", message)
        self.assertNotIn("matches no network header", message)

    def test_a_resolved_header_the_msa_lacks_is_named(self):
        message = self.report("WP_0123.10")
        self.assertIn(
            "WARNING: Reference ID 'WP_0123.10' resolves to WP_0123.10_protein_B, which "
            "the MSA lacks, so positions will be numbered by occupancy and the offset "
            "ignored.",
            message,
        )

    def test_an_unmatched_reference_leaves_numbering_inactive(self):
        message = self.report("XYZ")
        self.assertIn(
            "WARNING: Reference ID 'XYZ' matches no network header, so reference "
            "numbering will be inactive.",
            message,
        )

    def test_subset_success_and_incomplete_msa_coverage_are_reported(self):
        message = self.report("")
        self.assertEqual(
            message,
            "SUCCESS: FASTA is a strict subset of HDF5.\n"
            "FASTA vs HDF5:\nMatched: 3 of 3 | Missing: 0"
            "\n\nWARNING: MSA coverage is incomplete.\n"
            "FASTA vs MSA:\nMatched: 2 of 3 network headers | Missing: 1\n"
            "Missing examples: WP_0123.10_protein_B\n"
            "Missing nodes remain plotted but are excluded from "
            "alignment-dependent analyses.",
        )

    def test_fasta_headers_absent_from_the_network_are_an_error(self):
        (pathlib.Path(self.folder) / "superset.fasta").write_text(
            "".join(f">{header}\n{sequence}\n" for header, sequence in self.RECORDS.items())
            + ">Extra seq\nMCCCCCCK\n",
            encoding="utf-8",
        )
        message = self.report("", fasta="superset.fasta")
        self.assertTrue(
            message.startswith(
                "ERROR: FASTA is NOT a subset of HDF5.\n"
                "FASTA vs HDF5:\nMatched: 3 of 4 | Missing: 1\n"
                "Missing examples: Extra_seq\n\n"
            ),
            message,
        )
        self.assertIn(
            "FASTA vs MSA:\nMatched: 2 of 4 network headers | Missing: 2\n"
            "Missing examples: WP_0123.10_protein_B, Extra_seq\n",
            message,
        )


class ConsistencyEmbeddingReportTests(_ReportFixture, unittest.TestCase):
    """The embedding section matches nodes by header and sequence, as Embedding_Manager does."""

    def embedding_section(self, records, *, model_name="test_model", network_model=None):
        folder = pathlib.Path(self.folder)
        write_embedding_file(folder / "set_embeddings.h5", records, model_name=model_name)
        if network_model is not None:
            with h5py.File(folder / "network.h5", "a") as hf:
                hf.attrs["model_name"] = network_model
        message = self.report("", embedding="set_embeddings.h5")
        return message.split("\n\n")[-1]

    def test_full_coverage_is_a_success_naming_the_model(self):
        section = self.embedding_section(list(self.RECORDS.items()))
        self.assertEqual(
            section,
            "SUCCESS: The embedding covers all FASTA nodes.\n"
            "FASTA vs embedding:\nMatched: 3 of 3 FASTA headers | Missing: 0 | "
            "Different sequence: 0\n"
            "Embedding model: test_model (width 3, float16).",
        )

    def test_missing_and_different_sequences_are_a_warning(self):
        records = [("WP_0123.1_protein_A", "MKTAYIAK"), ("Other_seq", "MAAAGGG")]
        section = self.embedding_section(records)
        self.assertEqual(
            section,
            "WARNING: Embedding coverage is incomplete.\n"
            "FASTA vs embedding:\nMatched: 1 of 3 FASTA headers | Missing: 1 | "
            "Different sequence: 1\n"
            "Missing examples: WP_0123.10_protein_B\n"
            "Examples with a different sequence: Other_seq\n"
            "Missing nodes, and nodes whose sequence differs, remain plotted but are "
            "excluded from embedding-dependent analyses.\n"
            "Embedding model: test_model (width 3, float16).",
        )

    def test_a_network_built_with_another_model_gets_a_note(self):
        section = self.embedding_section(
            list(self.RECORDS.items()), model_name="esmc_6b", network_model="ankh_base"
        )
        self.assertIn(
            "NOTE: The network was built with ankh_base, not esmc_6b. The Viewer accepts "
            "this; embedding commands use the embedding's model.",
            section,
        )
        same = self.embedding_section(
            list(self.RECORDS.items()), model_name="esmc_6b", network_model="esmc_6b"
        )
        self.assertNotIn("NOTE", same)
        # A BLAST network was built from no embedding, so there is nothing to compare.
        blast = self.embedding_section(
            list(self.RECORDS.items()), model_name="esmc_6b", network_model="BLAST"
        )
        self.assertNotIn("NOTE", blast)

    def test_a_file_the_viewer_would_refuse_is_an_error(self):
        folder = pathlib.Path(self.folder)
        write_embedding_file(folder / "set_embeddings.h5", list(self.RECORDS.items()),
                             complete=False)
        message = self.report("", embedding="set_embeddings.h5")
        self.assertTrue(message.split("\n\n")[-1].startswith(
            "ERROR: The embedding file cannot be used, so the Viewer will not open with "
            "it: Embedding generation is incomplete."
        ), message)

    def test_no_embedding_adds_no_section(self):
        self.assertNotIn("embedding", self.report("").lower())


if __name__ == "__main__":
    unittest.main()
