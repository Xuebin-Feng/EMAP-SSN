"""Sparse_MSA_Converter: aligned FASTA to the sparse HDF5 MSA format.

The converter shares the MSA sanitizer, writes the matrix the sparse loader
reads back, moves the source FASTA into Full_Alignments, and its main()
reports failures through the exit code.
"""

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import h5py


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager
from tools import Sparse_MSA_Converter
from utilities.Sequence_Utils import AA_TO_INT


def write_fasta(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


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

    def test_sparse_loader_reads_back_the_converted_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "round_trip.fasta")
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(">a\nAC-.\n>b\n-c--\n")
            output_path = os.path.join(directory, "round_trip_sparse.h5")
            with redirect_stdout(io.StringIO()):
                succeeded = Sparse_MSA_Converter.build_sparse_alignment(path)
                loader = Alignment_Manager.SparseAlignmentLoader(output_path)
            with h5py.File(output_path, "r") as hf:
                matrix_shape = tuple(hf["matrix"].attrs["shape"])
                header_map = json.loads(hf["header_map"][()])
                aa_map = json.loads(hf["aa_map"][()])
            moved = os.path.join(directory, "Full_Alignments", "round_trip.fasta")
            self.assertFalse(os.path.exists(path))
            with open(moved, "rb") as handle:
                self.assertEqual(handle.read(), b">a\nAC-.\n>b\n-c--\n")

        self.assertTrue(succeeded)
        self.assertEqual([str(record.seq) for record in loader], ["AC--", "-C--"])
        self.assertEqual(loader.headers, ["a", "b"])
        # The last column holds only gaps, so no sparse entry records it.
        self.assertEqual(matrix_shape, (2, 4))
        self.assertEqual(loader.matrix.shape, (2, 4))
        self.assertEqual(header_map, {"a": 0, "b": 1})
        self.assertEqual(aa_map, AA_TO_INT)


class SparseMSAConverterExitCodeTests(unittest.TestCase):
    """main() must report failure so MCP jobs and terminals see non-zero."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        self.msa_dir = os.path.join(self.root, "msa")
        os.makedirs(self.msa_dir)
        # Point the import-time loader at a missing file so the developer's
        # tools_settings.json cannot leak into the fresh module, and restore
        # the SSN_TOOL_SETTINGS_* variables that main() exports.
        environment = mock.patch.dict(
            os.environ,
            {
                "SSN_TOOL_SETTINGS_SCRIPT": "Sparse_MSA_Converter.py",
                "SSN_TOOL_SETTINGS_FILE": os.path.join(self.root, "absent.json"),
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        module_path = os.path.join(SRC_DIR, "tools", "Sparse_MSA_Converter.py")
        spec = importlib.util.spec_from_file_location(
            "exit_code_sparse_converter", module_path
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def run_main(self, *, convert_all, input_fasta=""):
        settings_path = os.path.join(self.root, "converter.json")
        with open(settings_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "DIRECTORIES": {"MSA_DIR": self.msa_dir},
                    "Sparse_MSA_Converter.py": {
                        "CONVERT_ALL": convert_all,
                        "INPUT_FASTA": input_fasta,
                    },
                },
                handle,
            )
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = self.module.main([settings_path])
        return exit_code, output.getvalue()

    def msa_path(self, name):
        return os.path.join(self.msa_dir, name)

    def sparse_path(self, name):
        return os.path.splitext(self.msa_path(name))[0] + "_sparse.h5"

    def test_single_file_success_returns_zero(self):
        write_fasta(self.msa_path("good.fasta"), [("one", "AC-"), ("two", "A-D")])

        exit_code, text = self.run_main(convert_all=False, input_fasta="good.fasta")

        self.assertEqual(exit_code, 0)
        self.assertIn("🎉 Success! Sparse alignment saved to:", text)
        self.assertTrue(os.path.exists(self.sparse_path("good.fasta")))

    def test_missing_single_input_returns_nonzero(self):
        name = "Foldtype_IV_ATAs_44000_[ankh_base]_alignment.fasta"

        exit_code, text = self.run_main(convert_all=False, input_fasta=name)

        self.assertEqual(exit_code, 1)
        self.assertIn(f"❌ Error: File not found: {self.msa_path(name)}", text)
        self.assertEqual(os.listdir(self.msa_dir), [])

    def test_rejected_single_input_returns_nonzero_and_keeps_input(self):
        write_fasta(self.msa_path("ragged.fasta"), [("one", "AC"), ("two", "A")])

        exit_code, text = self.run_main(convert_all=False, input_fasta="ragged.fasta")

        self.assertEqual(exit_code, 1)
        self.assertIn("❌ MSA rejected:", text)
        self.assertTrue(os.path.exists(self.msa_path("ragged.fasta")))
        self.assertFalse(os.path.exists(self.sparse_path("ragged.fasta")))

    def test_hdf5_save_failure_returns_nonzero_and_leaves_no_partial_output(self):
        write_fasta(self.msa_path("good.fasta"), [("one", "AC-"), ("two", "A-D")])

        with mock.patch.object(
            self.module.h5py, "File", side_effect=OSError("disk full")
        ):
            exit_code, text = self.run_main(
                convert_all=False, input_fasta="good.fasta"
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("❌ Error during HDF5 save or file transfer: disk full", text)
        self.assertEqual(os.listdir(self.msa_dir), ["good.fasta"])

    def test_batch_with_every_file_converted_returns_zero(self):
        write_fasta(self.msa_path("first.fasta"), [("one", "AC-"), ("two", "A-D")])
        write_fasta(self.msa_path("second.fasta"), [("three", "MK"), ("four", "M-")])

        exit_code, text = self.run_main(convert_all=True)

        self.assertEqual(exit_code, 0)
        self.assertIn("🚀 Starting batch conversion of 2 alignments...", text)
        self.assertIn("✅ Batch conversion complete.", text)
        for name in ("first.fasta", "second.fasta"):
            self.assertTrue(os.path.exists(self.sparse_path(name)))

    def test_batch_with_one_failed_file_returns_nonzero_after_converting_rest(self):
        write_fasta(self.msa_path("good.fasta"), [("one", "AC-"), ("two", "A-D")])
        write_fasta(self.msa_path("ragged.fasta"), [("three", "AC"), ("four", "A")])

        exit_code, text = self.run_main(convert_all=True)

        self.assertEqual(exit_code, 1)
        self.assertIn("❌ MSA rejected:", text)
        self.assertIn("✅ Batch conversion complete.", text)
        self.assertTrue(os.path.exists(self.sparse_path("good.fasta")))
        self.assertTrue(os.path.exists(self.msa_path("ragged.fasta")))
        self.assertFalse(os.path.exists(self.sparse_path("ragged.fasta")))

    def test_empty_batch_returns_nonzero(self):
        exit_code, text = self.run_main(convert_all=True)

        self.assertEqual(exit_code, 1)
        self.assertIn(f"⚠️ No FASTA files found in {self.msa_dir} to convert.", text)
        self.assertNotIn("Batch conversion complete", text)


if __name__ == "__main__":
    unittest.main()
