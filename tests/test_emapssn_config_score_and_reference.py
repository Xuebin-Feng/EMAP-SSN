"""Configuration window: score-mode defaults and the consistency check's reference report."""

import os
import pathlib
import runpy
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_WINDOW = None


def setUpModule():
    global _WINDOW
    from utilities import Hardware_Acceleration as Hardware_Utils  # noqa: F401 - load torch before PySide6
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    with mock.patch.object(QApplication, "exec", return_value=0), mock.patch.object(
        sys, "exit", return_value=None
    ):
        namespace = runpy.run_path(str(SRC / "EMAPSSN_Config.py"), run_name="__main__")

    _WINDOW = namespace["window"]
    _WINDOW._cache_hash_request_id += 1
    for worker in _WINDOW._cache_hash_workers.values():
        worker.requestInterruption()
        worker.wait()
    _WINDOW._cache_hash_workers.clear()


def tearDownModule():
    from PySide6.QtWidgets import QApplication
    _WINDOW.close()
    QApplication.instance().processEvents()


def _items(combo):
    return [combo.itemText(index) for index in range(combo.count())]


class NormalizationModeRefreshTests(unittest.TestCase):
    def test_leaving_blast_restores_global_options_and_the_default(self):
        """Signals are blocked while the controls reset, so the list must be refreshed
        explicitly; a list left over from local mode silently rejected the default."""
        window = _WINDOW
        window._set_network_type_controls("alignment")
        window.cb_score_mode.setCurrentText("local")
        self.assertNotIn("alignment_length", _items(window.cb_norm_mode))

        window._set_network_type_controls("blast")
        self.assertEqual(window.cb_score_mode.currentIndex(), -1)
        self.assertEqual(window.cb_norm_mode.currentIndex(), -1)

        window._set_network_type_controls("alignment")
        self.assertEqual(window.cb_score_mode.currentText(), "global")
        self.assertIn("alignment_length", _items(window.cb_norm_mode))
        self.assertEqual(window.cb_norm_mode.currentText(), "alignment_length")
        self.assertFalse(window.cb_score_mode.signalsBlocked())
        self.assertFalse(window.cb_norm_mode.signalsBlocked())


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


class ConsistencyReferenceReportTests(unittest.TestCase):
    """The report resolves the reference as the Viewer does, over the network headers."""

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

    def report(self, reference):
        panel = _Panel()
        stand_in = SimpleNamespace(
            cb_fasta=_Choice("nodes.fasta"),
            cb_hdf5=_Choice("network.h5"),
            cb_msa=_Choice("alignment.fasta"),
            line_ref=_Choice(reference),
            tip_panel=panel,
            _resolved_directory_input=lambda key: self.folder,
        )
        type(_WINDOW).run_consistency_check(stand_in)
        self.assertNotIn("Error during consistency check", panel.message)
        return panel.message

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


if __name__ == "__main__":
    unittest.main()
