"""Network_Extraction: filtering a network to the headers of a FASTA subset."""

import codecs
import importlib.util
import io
import itertools
import os
import pathlib
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import h5py
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests import MISSING_TOOL_SETTINGS
from utilities.BLAST_Tabular import load_header_manifest
from utilities.Sequence_Utils import load_sanitized_fasta

# A private copy keeps the globals set below out of tools.Network_Extraction.
# Loading by path bypasses the tests package's import hook, so the copy's
# import-time settings loader is pointed at a missing file here instead.
MODULE_PATH = SRC_DIR / "tools" / "Network_Extraction.py"
SPEC = importlib.util.spec_from_file_location("network_extraction", MODULE_PATH)
network_extraction = importlib.util.module_from_spec(SPEC)
with mock.patch.dict(os.environ, {
    "SSN_TOOL_SETTINGS_SCRIPT": "Network_Extraction.py",
    "SSN_TOOL_SETTINGS_FILE": MISSING_TOOL_SETTINGS,
}):
    SPEC.loader.exec_module(network_extraction)


def _write_headers(handle, headers):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset(
        "headers",
        data=np.array(headers, dtype=object),
        dtype=string_dtype,
    )


def _write_complete_network(path, headers, *, blast):
    """Write a BLAST or alignment network with an edge between every pair."""
    pairs = list(itertools.combinations(range(len(headers)), 2))
    with h5py.File(path, "w") as handle:
        handle.attrs["model_name"] = "BLAST" if blast else "esm2"
        _write_headers(handle, headers)
        handle.create_dataset("i", data=np.array([i for i, _ in pairs]))
        handle.create_dataset("j", data=np.array([j for _, j in pairs]))
        if blast:
            handle.create_dataset("score", data=np.full(len(pairs), 1e-5))
        else:
            handle.create_dataset("seq_lens", data=np.full(len(headers), 10))
            for name in ("l_score", "l_len", "g_score", "g_len"):
                handle.create_dataset(name, data=np.ones(len(pairs)))


class NetworkExtractionTests(unittest.TestCase):
    @staticmethod
    def _filter(input_network, whitelist, output_network):
        report = io.StringIO()
        with redirect_stdout(report):
            network_extraction.filter_network(
                str(input_network),
                str(whitelist),
                str(output_network),
            )
        return report.getvalue()

    @staticmethod
    def _kept_headers(output_network):
        with h5py.File(output_network, "r") as output:
            return output["headers"].asstr()[:].tolist()

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

            # Imported DIAMOND networks keep model_name blast but carry their
            # own file's [DIAMOND] label; other searches keep the model name.
            for network_type, search_program, expected_name in (
                ("alignment", None, "selected_[esm2]_network.h5"),
                ("blast", None, "selected_[blast]_EValue.h5"),
                ("blast", "BLASTP", "selected_[blast]_EValue.h5"),
                ("blast", "DIAMOND", "selected_[DIAMOND]_EValue.h5"),
            ):
                with self.subTest(network_type=network_type, search_program=search_program):
                    input_network = root / f"{network_type}.h5"
                    if network_type == "alignment":
                        self._write_alignment_network(input_network)
                    else:
                        self._write_blast_network(input_network)
                    if search_program is not None:
                        with h5py.File(input_network, "a") as handle:
                            handle.attrs["search_program"] = search_program

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

    def test_whitelist_saved_with_a_byte_order_mark_keeps_its_first_record(self):
        # Windows Notepad can save UTF-8 with a BOM. Decoded as plain UTF-8 it
        # stayed on the first line, so the first header was never read.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            input_network = root / "master_network.h5"
            output_network = root / "subset_network.h5"
            whitelist = root / "subset.fasta"
            self._write_alignment_network(input_network)
            whitelist.write_bytes(codecs.BOM_UTF8 + b">A\nAAAA\n>C\nCCCC\n")

            report = self._filter(input_network, whitelist, output_network)

            self.assertEqual(self._kept_headers(output_network), ["A", "C"])
            self.assertIn("-> Found 2 sequences in filtered FASTA.", report)
            self.assertIn(
                "-> 0 of 2 whitelist headers matched no network header.", report
            )

    def test_whitelist_headers_are_sanitized_like_network_headers(self):
        # Network writers store FASTA headers sanitized: embedding networks
        # through load_sanitized_fasta, BLAST networks through the importer's
        # header manifest. A whitelist cut from the same raw FASTA must select
        # the same nodes.
        records = [
            ("sp|P69905|HBA_HUMAN Hemoglobin  subunit\talpha", "MVLSPADKTNVKAAWGKVGAHAGEYG"),
            ("WP_0123.1 putative [kinase] {fragment}", "MKKLLPTAAAGLLLLAAQPAMA"),
            ("tr|Q8XYZ1|ECOLI 3/4-dioxygenase #2", "MSTNPKPQRKTKRNTNRRPQDVKFPGG"),
            ("plain_id", "MAAAAGGGGLLLLWWW"),
        ]
        selected = records[:3]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            master_fasta = root / "master.fasta"
            whitelist = root / "subset.fasta"
            master_fasta.write_text(
                "".join(f">{header}\n{sequence}\n" for header, sequence in records),
                encoding="utf-8",
            )
            whitelist.write_text(
                "".join(f">{header}\n{sequence}\n" for header, sequence in selected),
                encoding="utf-8",
            )
            producers = {
                "alignment": (
                    lambda: load_sanitized_fasta(str(master_fasta), report=False)[0],
                    False,
                ),
                "blast": (
                    lambda: list(load_header_manifest(str(master_fasta)).headers),
                    True,
                ),
            }

            for network_type, (read_network_headers, blast) in producers.items():
                with self.subTest(network_type=network_type):
                    network_headers = read_network_headers()
                    for raw_header, _ in selected:
                        self.assertNotIn(raw_header, network_headers)
                    input_network = root / f"{network_type}.h5"
                    output_network = root / f"{network_type}_subset.h5"
                    _write_complete_network(input_network, network_headers, blast=blast)

                    report = self._filter(input_network, whitelist, output_network)

                    self.assertEqual(
                        self._kept_headers(output_network),
                        network_headers[:3],
                    )
                    self.assertIn(
                        "-> 0 of 3 whitelist headers matched no network header.",
                        report,
                    )

    def test_unmatched_whitelist_headers_are_counted_and_listed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            input_network = root / "master_EValue.h5"
            output_network = root / "subset_EValue.h5"
            whitelist = root / "subset.fasta"
            self._write_blast_network(input_network)
            whitelist.write_text(
                ">A first sequence\nAAAA\n>C\nCCCC\n>Z\nZZZZ\n",
                encoding="utf-8",
            )

            report = self._filter(input_network, whitelist, output_network)

            self.assertEqual(self._kept_headers(output_network), ["C"])
            self.assertIn("-> Found 3 sequences in filtered FASTA.", report)
            self.assertIn(
                "-> 2 of 3 whitelist headers matched no network header.", report
            )
            self.assertEqual(
                re.findall(r"^    - (.*)$", report, re.MULTILINE),
                ["A_first_sequence", "Z"],
            )

    def test_whitelist_records_are_matched_by_header_alone(self):
        # Sanitizing the whitelist as a whole FASTA (load_sanitized_fasta)
        # would drop header-only records, merge records sharing a sequence
        # (BLAST networks keep both), and number repeated headers by their
        # position in the subset rather than in the network's own FASTA.
        cases = (
            # (label, network headers, BLAST network, whitelist, kept, unmatched)
            ("header-only list", ["A", "B", "C"], False, ">A\n>C\n", ["A", "C"], 0),
            (
                "records sharing a sequence",
                ["P", "Q", "R"],
                True,
                ">P\nMKV\n>Q\nMKV\n",
                ["P", "Q"],
                0,
            ),
            (
                "repeated header",
                ["X_1", "X_2", "X_3", "Y"],
                False,
                ">X\nTTT\n>X\nUUU\n>Y\nVVV\n",
                ["Y"],
                1,
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            for index, case in enumerate(cases):
                label, network_headers, blast, text, kept, unmatched = case
                with self.subTest(label):
                    input_network = root / f"network_{index}.h5"
                    output_network = root / f"subset_{index}.h5"
                    whitelist = root / f"whitelist_{index}.fasta"
                    _write_complete_network(input_network, network_headers, blast=blast)
                    whitelist.write_text(text, encoding="utf-8")

                    report = self._filter(input_network, whitelist, output_network)

                    self.assertEqual(self._kept_headers(output_network), kept)
                    self.assertIn(
                        f"-> {unmatched} of 2 whitelist headers matched no network "
                        "header.",
                        report,
                    )


if __name__ == "__main__":
    unittest.main()
