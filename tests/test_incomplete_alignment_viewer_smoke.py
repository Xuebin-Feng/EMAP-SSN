"""Offscreen integration smoke test for MainViewer.load_global_alignment."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import EMAPSSN_Config as cfg
from EMAPSSN_Viewer import MainViewer


def write_fasta(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


class IncompleteAlignmentViewerSmokeTests(unittest.TestCase):
    def test_viewer_load_method_accepts_partial_and_zero_coverage(self):
        old_msa = cfg.MSA_FILE
        try:
            with tempfile.TemporaryDirectory() as directory:
                for filename, records, expected_count in [
                    ("partial.fasta", [("node1", "AC")], 1),
                    ("zero.fasta", [("other", "AC")], 0),
                ]:
                    msa_path = os.path.join(directory, filename)
                    write_fasta(msa_path, records)
                    cfg.MSA_FILE = msa_path

                    viewer = MainViewer.__new__(MainViewer)
                    viewer.full_headers = ["node1", "node2"]
                    viewer.active_reference = "node1"
                    viewer.alignment_offset = 0
                    with redirect_stdout(io.StringIO()):
                        viewer.load_global_alignment()

                    self.assertIsNotNone(viewer.alignment.aln)
                    self.assertEqual(len(viewer.alignment.aln), expected_count)
        finally:
            cfg.MSA_FILE = old_msa


if __name__ == "__main__":
    unittest.main()
