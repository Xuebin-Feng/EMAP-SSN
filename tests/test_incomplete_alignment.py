"""Regression coverage for incomplete MSA loading and AA expression semantics."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager
import Command_Engine


class TTYBuffer(io.StringIO):
    def isatty(self):
        return True


def write_fasta(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


class IncompleteAlignmentLoaderTests(unittest.TestCase):
    def test_partial_fasta_load_tracks_exact_coverage_and_warns_red(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta_path = os.path.join(directory, "partial.fasta")
            write_fasta(
                fasta_path,
                [("node1 full", "AC"), ("node3 full", "TC")],
            )
            output = TTYBuffer()
            with redirect_stdout(output):
                manager = Alignment_Manager.Alignment_Manager(
                    fasta_path,
                    full_headers=["node1 full", "node2 full", "node3 full"],
                    active_reference="node1",
                )

        self.assertEqual(manager.matched_headers, ["node1 full", "node3 full"])
        self.assertEqual(manager.missing_headers, ["node2 full"])
        np.testing.assert_array_equal(manager.viewer_to_aln, np.array([0, -1, 1]))
        np.testing.assert_array_equal(
            manager.aligned_node_mask,
            np.array([True, False, True]),
        )
        self.assertTrue(manager.has_reference)
        warning = output.getvalue()
        self.assertIn("\033[91mWARNING: Incomplete MSA coverage", warning)
        self.assertIn("Aligned network nodes: 2/3 (66.7%)", warning)
        self.assertIn("  - node2 full", warning)
        self.assertIn("\033[0m", warning)

    def test_exact_full_header_matching_does_not_accept_shortened_description(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta_path = os.path.join(directory, "shortened.fasta")
            write_fasta(fasta_path, [("node1", "AC")])
            with redirect_stdout(io.StringIO()):
                manager = Alignment_Manager.Alignment_Manager(
                    fasta_path,
                    full_headers=["node1 full description"],
                )

        self.assertEqual(len(manager.aln), 0)
        self.assertEqual(manager.matched_headers, [])
        self.assertEqual(manager.missing_headers, ["node1 full description"])

    def test_zero_overlap_is_a_loaded_empty_alignment_state(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta_path = os.path.join(directory, "zero.fasta")
            write_fasta(fasta_path, [("other", "AC")])
            with redirect_stdout(io.StringIO()) as output:
                manager = Alignment_Manager.Alignment_Manager(
                    fasta_path,
                    full_headers=["node1", "node2"],
                    active_reference="node1",
                    alignment_offset=10,
                )

        self.assertIsNotNone(manager.aln)
        self.assertEqual(len(manager.aln), 0)
        self.assertEqual(manager.valid_cols, set())
        self.assertFalse(manager.has_reference)
        self.assertEqual(manager.offset, 0)
        np.testing.assert_array_equal(manager.viewer_to_aln, np.array([-1, -1]))
        self.assertIn("Aligned network nodes: 0/2 (0.0%)", output.getvalue())
        self.assertIn("pure occupancy mode", output.getvalue())

    def test_missing_reference_falls_back_without_disabling_partial_msa(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta_path = os.path.join(directory, "missing_ref.fasta")
            write_fasta(fasta_path, [("node1", "AC"), ("node3", "TC")])
            with redirect_stdout(io.StringIO()):
                manager = Alignment_Manager.Alignment_Manager(
                    fasta_path,
                    full_headers=["node1", "node2", "node3"],
                    active_reference="node2",
                    alignment_offset=10,
                )

        self.assertIsNotNone(manager.aln)
        self.assertEqual(len(manager.aln), 2)
        self.assertFalse(manager.has_reference)
        self.assertEqual(manager.offset, 0)
        self.assertEqual(manager.resolved_ref_full, "None")
        self.assertTrue(manager.col_to_label)

    def test_sparse_hdf5_partial_load_preserves_extra_row_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            h5_path = os.path.join(directory, "partial.h5")
            with h5py.File(h5_path, "w") as hf:
                matrix = hf.create_group("matrix")
                matrix.attrs["shape"] = (2, 2)
                matrix.create_dataset("data", data=np.array([1, 2, 2, 1], dtype=np.uint8))
                matrix.create_dataset("indices", data=np.array([0, 1, 0, 1], dtype=np.int32))
                matrix.create_dataset("indptr", data=np.array([0, 2, 4], dtype=np.int32))
                hf.create_dataset("headers", data=np.array([b"node1", b"extra"]))
                hf.create_dataset("int_to_aa", data=json.dumps({"1": "A", "2": "C"}).encode("utf-8"))

            with redirect_stdout(io.StringIO()):
                manager = Alignment_Manager.Alignment_Manager(
                    h5_path,
                    full_headers=["node1", "node2"],
                )

        self.assertEqual(len(manager.aln), 1)
        self.assertEqual(manager.aln.headers, ["node1"])
        self.assertEqual(manager.missing_headers, ["node2"])
        np.testing.assert_array_equal(manager.viewer_to_aln, np.array([0, -1]))

    def test_warning_lists_ten_examples_and_reports_omitted_count(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta_path = os.path.join(directory, "truncated_warning.fasta")
            write_fasta(fasta_path, [("matched", "AC")])
            headers = ["matched"] + [f"missing_{i}" for i in range(12)]
            with redirect_stdout(io.StringIO()) as output:
                Alignment_Manager.Alignment_Manager(
                    fasta_path,
                    full_headers=headers,
                )

        warning = output.getvalue()
        self.assertIn("showing 10 of 12", warning)
        self.assertIn("  - missing_9", warning)
        self.assertNotIn("  - missing_10", warning)
        self.assertIn("... and 2 more.", warning)


class ThreeStateExpressionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        fasta_path = os.path.join(self.temp_dir.name, "partial.fasta")
        write_fasta(fasta_path, [("node1", "AC"), ("node3", "TC")])
        with redirect_stdout(io.StringIO()):
            self.alignment = Alignment_Manager.Alignment_Manager(
                fasta_path,
                full_headers=["node1", "node2", "node3"],
                active_reference="node1",
            )
        self.headers = ["node1", "node2", "node3"]
        self.viewer = SimpleNamespace(
            full_headers=self.headers,
            alignment=self.alignment,
        )
        self.viewer_to_aln, self.valid_indices = Command_Engine.get_alignment_mapping(
            self.viewer
        )
        self.metadata = {
            "Length": {
                "type": "number",
                "values": np.array([600.0, 600.0, 600.0]),
            }
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def evaluate(self, expression):
        return Command_Engine.parse_advanced_expression(
            expression,
            self.viewer_to_aln,
            self.valid_indices,
            self.headers,
            alignment=self.alignment,
            metadata=self.metadata,
        )

    def test_aa_and_negated_aa_are_both_false_for_unaligned_nodes(self):
        np.testing.assert_array_equal(self.evaluate("A1"), [True, False, False])
        np.testing.assert_array_equal(self.evaluate("!A1"), [False, False, True])

    def test_independent_known_or_clause_can_match_unaligned_node(self):
        np.testing.assert_array_equal(
            self.evaluate("A1|{Length>500}"),
            [True, True, True],
        )

    def test_unknown_propagates_through_and_xor_and_parenthesized_not(self):
        np.testing.assert_array_equal(
            self.evaluate("A1&{Length>500}"),
            [True, False, False],
        )
        np.testing.assert_array_equal(
            self.evaluate("A1^{Length>500}"),
            [False, False, True],
        )
        np.testing.assert_array_equal(
            self.evaluate("!(A1|{Length<500})"),
            [False, False, True],
        )

    def test_non_aa_expression_still_matches_unaligned_nodes(self):
        np.testing.assert_array_equal(
            self.evaluate("{Length>500}"),
            [True, True, True],
        )


if __name__ == "__main__":
    unittest.main()
