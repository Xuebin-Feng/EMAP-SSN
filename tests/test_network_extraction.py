import importlib.util
import io
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import h5py
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODULE_PATH = SRC_DIR / "tools" / "Network_Extraction.py"
SPEC = importlib.util.spec_from_file_location("network_extraction", MODULE_PATH)
network_extraction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(network_extraction)


def _write_headers(handle, headers):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset(
        "headers",
        data=np.array(headers, dtype=object),
        dtype=string_dtype,
    )


class NetworkExtractionTests(unittest.TestCase):
    def _write_alignment_network(self, path):
        with h5py.File(path, "w") as handle:
            handle.attrs["model_name"] = "esm2"
            handle.attrs["sentinel"] = "preserved"
            _write_headers(handle, ["A", "B", "C"])
            handle.create_dataset("seq_lens", data=np.array([10, 20, 30]))
            handle.create_dataset("i", data=np.array([0, 0, 1, 2]))
            handle.create_dataset("j", data=np.array([1, 2, 2, 0]))
            handle.create_dataset("l_score", data=np.array([1, 2, 3, 4]))
            handle.create_dataset("l_len", data=np.array([11, 12, 13, 14]))
            handle.create_dataset("g_score", data=np.array([21, 22, 23, 24]))
            handle.create_dataset("g_len", data=np.array([31, 32, 33, 34]))

    def _write_blast_network(self, path):
        with h5py.File(path, "w") as handle:
            handle.attrs["model_name"] = "blast"
            _write_headers(handle, ["A", "B", "C"])
            handle.create_dataset("i", data=np.array([0, 0, 1, 2]))
            handle.create_dataset("j", data=np.array([1, 2, 2, 0]))
            handle.create_dataset("score", data=np.array([1e-2, 1e-4, 1e-6, 1e-8]))

    def test_alignment_network_is_filtered_without_path_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            input_network = root / "master_network.h5"
            output_network = root / "subset_[esm2]_network.h5"
            whitelist = root / "subset.fasta"
            legacy_paths = root / "master_paths.h5"
            self._write_alignment_network(input_network)
            whitelist.write_text(">A\nAAAA\n>C\nCCCC\n", encoding="utf-8")
            legacy_paths.write_bytes(b"legacy path artifact")

            with redirect_stdout(io.StringIO()):
                network_extraction.filter_network(
                    str(input_network),
                    str(whitelist),
                    str(output_network),
                )

            with h5py.File(output_network, "r") as output:
                self.assertEqual(output.attrs["sentinel"], "preserved")
                self.assertEqual(output["headers"].asstr()[:].tolist(), ["A", "C"])
                np.testing.assert_array_equal(output["seq_lens"][:], [10, 30])
                np.testing.assert_array_equal(output["i"][:], [0, 1])
                np.testing.assert_array_equal(output["j"][:], [1, 0])
                np.testing.assert_array_equal(output["l_score"][:], [2, 4])
                np.testing.assert_array_equal(output["l_len"][:], [12, 14])
                np.testing.assert_array_equal(output["g_score"][:], [22, 24])
                np.testing.assert_array_equal(output["g_len"][:], [32, 34])

            self.assertEqual(legacy_paths.read_bytes(), b"legacy path artifact")
            self.assertEqual(
                sorted(path.name for path in root.glob("*_paths.h5")),
                ["master_paths.h5"],
            )

    def test_blast_network_is_filtered_without_path_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            input_network = root / "master_EValue.h5"
            output_network = root / "subset_[blast]_EValue.h5"
            whitelist = root / "subset.fasta"
            self._write_blast_network(input_network)
            whitelist.write_text(">A\nAAAA\n>C\nCCCC\n", encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                network_extraction.filter_network(
                    str(input_network),
                    str(whitelist),
                    str(output_network),
                )

            with h5py.File(output_network, "r") as output:
                self.assertEqual(output["headers"].asstr()[:].tolist(), ["A", "C"])
                np.testing.assert_array_equal(output["i"][:], [0, 1])
                np.testing.assert_array_equal(output["j"][:], [1, 0])
                np.testing.assert_array_equal(output["score"][:], [1e-4, 1e-8])
                self.assertNotIn("seq_lens", output)

            self.assertEqual(list(root.glob("*_paths.h5")), [])

    def test_runtime_output_names_preserve_network_schema_conventions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            whitelist = root / "selected.fasta"
            whitelist.write_text(">A\nAAAA\n", encoding="utf-8")

            for network_type, expected_name in (
                ("alignment", "selected_[esm2]_network.h5"),
                ("blast", "selected_[blast]_EValue.h5"),
            ):
                with self.subTest(network_type=network_type):
                    input_network = root / f"{network_type}.h5"
                    if network_type == "alignment":
                        self._write_alignment_network(input_network)
                    else:
                        self._write_blast_network(input_network)

                    network_extraction.INPUT_NET = str(input_network)
                    network_extraction.INPUT_FASTA = str(whitelist)
                    network_extraction.NETWORK_DIR = str(root)
                    network_extraction.FASTA_DIR = str(root)
                    network_extraction.configure_runtime_paths()

                    self.assertEqual(
                        network_extraction.OUTPUT_NET,
                        os.path.join(temp_dir, expected_name),
                    )

        self.assertFalse(hasattr(network_extraction, "PATH_DIR"))
        self.assertFalse(hasattr(network_extraction, "INPUT_PATHS"))
        self.assertFalse(hasattr(network_extraction, "OUTPUT_PATHS"))


if __name__ == "__main__":
    unittest.main()
