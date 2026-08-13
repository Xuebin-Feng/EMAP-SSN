"""Regression coverage for secure, position-preserving MSA loading."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager
from tools import Sparse_MSA_Converter
from utilities.MSA_Sanitization import AA_TO_INT, INT_TO_AA


def write_fasta(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def write_sparse_h5(path, *, shape, data, indices, indptr, headers, mapping):
    with h5py.File(path, "w") as hf:
        matrix = hf.create_group("matrix")
        matrix.attrs["shape"] = shape
        matrix.create_dataset("data", data=np.asarray(data))
        matrix.create_dataset("indices", data=np.asarray(indices, dtype=np.int32))
        matrix.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int32))
        string_dtype = h5py.string_dtype(encoding="utf-8")
        matrix_headers = np.asarray(headers, dtype=object)
        hf.create_dataset("headers", data=matrix_headers, dtype=string_dtype)
        hf.create_dataset("int_to_aa", data=json.dumps(mapping))


class MSAFastaSanitizationTests(unittest.TestCase):
    def test_clean_alignment_is_silent_and_preserves_extended_residues(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "clean.fasta")
            write_fasta(path, [("one", "ABZJUO-"), ("two", "ABZJUO-")])
            output = io.StringIO()
            with redirect_stdout(output):
                loader = Alignment_Manager.InMemorySparseLoader(path)

        self.assertNotIn("MSA sanitization was applied", output.getvalue())
        self.assertEqual(str(loader[0].seq), "ABZJUO-")
        for residue in "BZJUO":
            self.assertEqual(loader.int_to_aa[AA_TO_INT[residue]], residue)
            self.assertGreater(AA_TO_INT[residue], 21)
        np.testing.assert_array_equal(
            loader.bulk_residue_check(1, "B"),
            np.array([True, True]),
        )
        self.assertEqual(loader.get_frequencies(1)[0], "B")

    def test_sanitization_preserves_columns_and_reports_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dirty.fasta")
            write_fasta(
                path,
                [("Alpha??__Beta", "ac.?*B"), ("Other", "AC---B")],
            )
            original = Path(path).read_bytes()
            output = io.StringIO()
            with redirect_stdout(output):
                manager = Alignment_Manager.Alignment_Manager(
                    path,
                    full_headers=["Alpha_Beta", "Other"],
                )
            after = Path(path).read_bytes()

        text = output.getvalue()
        self.assertEqual(text.count("MSA sanitization was applied"), 1)
        self.assertIn("Headers modified: 1", text)
        self.assertIn("Residues uppercased: 2", text)
        self.assertIn("Gap symbols normalized: 1", text)
        self.assertIn("Illegal residues replaced with X: 2", text)
        self.assertIn("Source file was not modified", text)
        self.assertEqual(original, after)
        self.assertEqual(manager.aln.headers, ["Alpha_Beta", "Other"])
        self.assertEqual(str(manager.aln[0].seq), "AC-XXB")
        self.assertEqual(manager.aln.matrix.shape, (2, 6))
        self.assertEqual(manager.aln.matrix[0, 4], AA_TO_INT["X"])

    def test_unequal_lengths_are_rejected_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "unequal.fasta")
            write_fasta(path, [("one", "AC-"), ("two", "AC")])
            original = Path(path).read_bytes()
            output = io.StringIO()
            with redirect_stdout(output):
                loader, is_sparse = Alignment_Manager.load_alignment_smart(path)

            self.assertEqual(Path(path).read_bytes(), original)

        self.assertIsNone(loader)
        self.assertFalse(is_sparse)
        self.assertIn("ERROR: MSA rejected", output.getvalue())
        self.assertIn("equal aligned lengths", output.getvalue())

    def test_header_sanitization_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "collision.fasta")
            write_fasta(path, [("A?", "AC"), ("A#", "AC")])
            with redirect_stdout(io.StringIO()) as output:
                loader, _ = Alignment_Manager.load_alignment_smart(path)

        self.assertIsNone(loader)
        self.assertIn("creates a duplicate header", output.getvalue())

    def test_complete_fasta_is_validated_before_header_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "filtered_invalid.fasta")
            write_fasta(path, [("kept", "AC"), ("excluded", "A")])
            with redirect_stdout(io.StringIO()) as output:
                loader, _ = Alignment_Manager.load_alignment_smart(
                    path,
                    filter_headers=["kept"],
                )

        self.assertIsNone(loader)
        self.assertIn("equal aligned lengths", output.getvalue())

    def test_sequence_before_first_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "malformed.fasta")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("AC\n>one\nAC\n")
            with redirect_stdout(io.StringIO()) as output:
                loader, _ = Alignment_Manager.load_alignment_smart(path)

        self.assertIsNone(loader)
        self.assertIn("before the first header", output.getvalue())


class SparseHDF5SanitizationTests(unittest.TestCase):
    def test_clean_legacy_hdf5_loads_without_sanitization_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "legacy.h5")
            write_sparse_h5(
                path,
                shape=(1, 2),
                data=np.array([1, 21], dtype=np.uint8),
                indices=[0, 1],
                indptr=[0, 2],
                headers=["one"],
                mapping={str(code): residue for code, residue in INT_TO_AA.items() if code <= 21},
            )
            output = io.StringIO()
            with redirect_stdout(output):
                loader = Alignment_Manager.SparseAlignmentLoader(path)

        self.assertEqual(str(loader[0].seq), "AX")
        self.assertNotIn("MSA sanitization was applied", output.getvalue())

    def test_hdf5_content_is_canonicalized_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dirty.h5")
            write_sparse_h5(
                path,
                shape=(1, 4),
                data=np.array([1, 30, 31, 0], dtype=np.int16),
                indices=[0, 1, 2, 3],
                indptr=[0, 4],
                headers=["bad?"],
                mapping={"1": "a", "30": "B", "31": "?"},
            )
            original = Path(path).read_bytes()
            output = io.StringIO()
            with redirect_stdout(output):
                loader = Alignment_Manager.SparseAlignmentLoader(path)
            after = Path(path).read_bytes()

        text = output.getvalue()
        self.assertEqual(text.count("MSA sanitization was applied"), 1)
        self.assertIn("Headers modified: 1", text)
        self.assertIn("Illegal residues replaced with X: 1", text)
        self.assertIn("Sparse gap/zero entries removed: 1", text)
        self.assertEqual(original, after)
        self.assertEqual(loader.headers, ["bad_"])
        self.assertEqual(str(loader[0].seq), "ABX-")
        self.assertEqual(loader.matrix[0, 1], AA_TO_INT["B"])

    def test_invalid_csr_structure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "duplicates.h5")
            write_sparse_h5(
                path,
                shape=(1, 2),
                data=np.array([1, 2], dtype=np.uint8),
                indices=[0, 0],
                indptr=[0, 2],
                headers=["one"],
                mapping={"1": "A", "2": "C"},
            )
            with redirect_stdout(io.StringIO()) as output:
                loader, _ = Alignment_Manager.load_alignment_smart(path)

        self.assertIsNone(loader)
        self.assertIn("duplicate column indices", output.getvalue())

    def test_external_required_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "target.h5")
            with h5py.File(target, "w") as hf:
                hf.create_group("matrix")
            path = os.path.join(directory, "external.h5")
            with h5py.File(path, "w") as hf:
                hf["matrix"] = h5py.ExternalLink(os.path.basename(target), "/matrix")
                hf.create_dataset("headers", data=np.array([b"one"]))
                hf.create_dataset("int_to_aa", data=json.dumps({"1": "A"}))
            with redirect_stdout(io.StringIO()) as output:
                loader, _ = Alignment_Manager.load_alignment_smart(path)

        self.assertIsNone(loader)
        self.assertIn("local hard link", output.getvalue())

    def test_missing_object_header_count_and_invalid_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = os.path.join(directory, "missing.h5")
            with h5py.File(missing_path, "w") as hf:
                hf.create_dataset("headers", data=np.array([b"one"]))
                hf.create_dataset("int_to_aa", data=json.dumps({"1": "A"}))

            header_path = os.path.join(directory, "headers.h5")
            write_sparse_h5(
                header_path,
                shape=(2, 1),
                data=np.array([1], dtype=np.uint8),
                indices=[0],
                indptr=[0, 1, 1],
                headers=["one"],
                mapping={"1": "A"},
            )

            json_path = os.path.join(directory, "json.h5")
            write_sparse_h5(
                json_path,
                shape=(1, 1),
                data=np.array([1], dtype=np.uint8),
                indices=[0],
                indptr=[0, 1],
                headers=["one"],
                mapping={"1": "A"},
            )
            with h5py.File(json_path, "r+") as hf:
                del hf["int_to_aa"]
                hf.create_dataset("int_to_aa", data="{")

            expected_messages = (
                (missing_path, "missing required object 'matrix'"),
                (header_path, "header count"),
                (json_path, "invalid JSON"),
            )
            for path, expected in expected_messages:
                with self.subTest(path=path):
                    with redirect_stdout(io.StringIO()) as output:
                        loader, _ = Alignment_Manager.load_alignment_smart(path)
                    self.assertIsNone(loader)
                    self.assertIn(expected, output.getvalue())

    def test_invalid_utf8_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "utf8.h5")
            with h5py.File(path, "w") as hf:
                matrix = hf.create_group("matrix")
                matrix.attrs["shape"] = (1, 1)
                matrix.create_dataset("data", data=np.array([1], dtype=np.uint8))
                matrix.create_dataset("indices", data=np.array([0], dtype=np.int32))
                matrix.create_dataset("indptr", data=np.array([0, 1], dtype=np.int32))
                hf.create_dataset("headers", data=np.array([b"\xff"], dtype="S1"))
                hf.create_dataset("int_to_aa", data=json.dumps({"1": "A"}))
            with redirect_stdout(io.StringIO()) as output:
                loader, _ = Alignment_Manager.load_alignment_smart(path)

        self.assertIsNone(loader)
        self.assertIn("not valid UTF-8", output.getvalue())


class SparseMSAConverterTests(unittest.TestCase):
    def test_converter_uses_shared_sanitizer_and_canonical_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "convert.fasta")
            write_fasta(path, [("bad?", "b."), ("other", "BX")])
            output_path = os.path.splitext(path)[0] + "_sparse.h5"
            output = io.StringIO()
            with redirect_stdout(output):
                succeeded = Sparse_MSA_Converter.build_sparse_alignment(path)

            self.assertTrue(succeeded)
            self.assertTrue(os.path.exists(output_path))
            self.assertFalse(os.path.exists(path))
            self.assertTrue(
                os.path.exists(
                    os.path.join(directory, "Full_Alignments", "convert.fasta")
                )
            )
            with h5py.File(output_path, "r") as hf:
                mapping = json.loads(hf["int_to_aa"][()])
                headers = hf["headers"].asstr()[:].tolist()
                data = hf["matrix/data"][:]

        self.assertIn("MSA sanitization was applied", output.getvalue())
        self.assertEqual(mapping[str(AA_TO_INT["B"])], "B")
        self.assertEqual(headers, ["bad_", "other"])
        self.assertIn(AA_TO_INT["B"], data)

    def test_converter_validation_failure_leaves_input_and_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "invalid.fasta")
            write_fasta(path, [("one", "AC"), ("two", "A")])
            output_path = os.path.splitext(path)[0] + "_sparse.h5"
            with redirect_stdout(io.StringIO()):
                succeeded = Sparse_MSA_Converter.build_sparse_alignment(path)

            self.assertFalse(succeeded)
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
