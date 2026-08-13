"""Command-level regressions for incomplete alignment coverage."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager
import SSN_Config as cfg
from commands import alignment as alignment_command
from commands import label as label_command
from commands import logo as logo_command
from commands import query as query_command
from commands import reference as reference_command
from commands import select as select_command


def write_fasta(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def load_manager(path, headers, reference=None):
    with redirect_stdout(io.StringIO()):
        return Alignment_Manager.Alignment_Manager(
            path,
            full_headers=headers,
            active_reference=reference,
        )


class IncompleteAlignmentCommandTests(unittest.TestCase):
    def test_alignment_command_accepts_zero_overlap_and_reports_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            msa_path = os.path.join(directory, "zero.fasta")
            write_fasta(msa_path, [("other", "AC")])
            headers = ["node1", "node2"]
            viewer = SimpleNamespace(
                full_headers=headers,
                active_reference="node1",
                alignment=SimpleNamespace(aln=None),
                console_text=SimpleNamespace(text=""),
            )

            def load_global_alignment():
                viewer.alignment = Alignment_Manager.Alignment_Manager(
                    cfg.MSA_FILE,
                    full_headers=headers,
                    active_reference=viewer.active_reference,
                )

            viewer.load_global_alignment = load_global_alignment
            old_msa = cfg.MSA_FILE
            try:
                with redirect_stdout(io.StringIO()) as output:
                    alignment_command.run(viewer, [msa_path])
            finally:
                cfg.MSA_FILE = old_msa

        self.assertIsNotNone(viewer.alignment.aln)
        self.assertEqual(len(viewer.alignment.aln), 0)
        self.assertIn("0/2 network nodes aligned", output.getvalue())
        self.assertIn("0/2 aligned", viewer.console_text.text)

    def test_reference_command_keeps_missing_reference_inactive(self):
        with tempfile.TemporaryDirectory() as directory:
            msa_path = os.path.join(directory, "partial.fasta")
            write_fasta(msa_path, [("node1", "AC")])
            headers = ["node1", "node2"]
            viewer = SimpleNamespace(
                full_headers=headers,
                active_reference="node1",
                resolved_ref_full="node1",
                console_text=SimpleNamespace(text=""),
            )
            viewer.alignment = load_manager(msa_path, headers, "node1")

            def load_global_alignment():
                viewer.alignment = Alignment_Manager.Alignment_Manager(
                    msa_path,
                    full_headers=headers,
                    active_reference=viewer.active_reference,
                )

            viewer.load_global_alignment = load_global_alignment
            with redirect_stdout(io.StringIO()):
                reference_command.run(viewer, ["node2"])

        self.assertEqual(viewer.active_reference, "node2")
        self.assertFalse(viewer.alignment.has_reference)
        self.assertIn("configured but inactive", viewer.console_text.text)

    def test_query_logo_and_label_reject_loaded_zero_coverage_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            msa_path = os.path.join(directory, "zero.fasta")
            write_fasta(msa_path, [("other", "AC")])
            alignment = load_manager(msa_path, ["node1"], "node1")
            viewer = SimpleNamespace(
                alignment=alignment,
                full_headers=["node1"],
                active_reference="node1",
                console_text=SimpleNamespace(text=""),
            )

            with redirect_stdout(io.StringIO()):
                query_command.run(viewer, ["[1]"])
            self.assertIn("no aligned rows", viewer.console_text.text)

            logo_command.run(viewer, ["[1]"])
            self.assertIn("no aligned rows", viewer.console_text.text)

            with redirect_stdout(io.StringIO()):
                label_command.run(viewer, [])
            self.assertIn("no aligned rows", viewer.console_text.text)

    def test_select_not_aa_excludes_unaligned_node(self):
        with tempfile.TemporaryDirectory() as directory:
            msa_path = os.path.join(directory, "partial.fasta")
            write_fasta(msa_path, [("node1", "AC"), ("node3", "TC")])
            headers = ["node1", "node2", "node3"]
            alignment = load_manager(msa_path, headers, "node1")
            viewer = SimpleNamespace(
                alignment=alignment,
                full_headers=headers,
                visible_mask=np.ones(3, dtype=bool),
                selected_indices=[],
                console_text=SimpleNamespace(text=""),
                update_selection_visual=lambda: None,
            )

            with mock.patch.object(cfg, "HEADER_LIST_DIR", directory):
                select_command.run(viewer, ["!A1"])

        self.assertEqual(viewer.selected_indices, [2])


if __name__ == "__main__":
    unittest.main()
