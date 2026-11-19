"""Embedding Import (src/tools/Embedding_Import.py), its Tools card and its MCP settings.

The tool converts per-residue embeddings computed elsewhere (an HDF5 file of one
dataset per protein, a .npz file or a folder of .npy files) into an embedding
database for a FASTA's sanitized records. The first classes run it on made-up
sources in temporary folders: how IDs are matched, which arrays are refused and
why, the sources it reads, and the output, report and resume. Then come main(),
the Saving Mode default shared with Generate Embeddings, the Tools card and the
MCP validation.
"""

import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, SRC / "tools"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import Embedding_Import
from utilities.HDF5_Storage import read_embedding_manifest

UNIPROT_FASTA = (
    ">sp|P00001|AAA_ECOLI Alpha protein OS=Escherichia coli\nMKTAY\n"
    ">sp|P00002|BBB_ECOLI Beta protein OS=Escherichia coli\nACDEFG\n"
)


def rows(count, width=4, dtype=np.float32, seed=42):
    return np.random.default_rng(seed + count).random((count, width)).astype(dtype)


class ImportCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.embed_dir = self.root / "Embeddings"

    def fasta(self, text, name="set.fasta"):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def h5_source(self, arrays, name="source.h5", original_ids=None):
        path = self.root / name
        with h5py.File(path, "w") as hf:
            for key, array in arrays.items():
                hf.create_dataset(key, data=array)
            for key, original in (original_ids or {}).items():
                hf[key].attrs["original_id"] = original
        return path

    def run_import(self, fasta, source, label="my_model", mode="float32", id_map=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return Embedding_Import.import_embeddings(
                str(fasta), str(source), label, mode, str(self.embed_dir),
                id_map_path=str(id_map) if id_map else None,
            )

    def read(self, path, complete=True):
        with h5py.File(path, "r") as hf:
            manifest = read_embedding_manifest(hf, require_complete=complete)
            arrays = {header: hf["embeddings"][header][()] for header in hf["embeddings"]}
            attrs = {key: hf.attrs[key] for key in hf.attrs}
        return manifest, arrays, attrs

    def report(self, result):
        lines = Path(result["report_path"]).read_text(encoding="utf-8").splitlines()
        return [line.split("\t") for line in lines[1:]]


class MatchingTests(ImportCase):
    def test_uniprot_headers_find_their_accessions_and_the_file_is_complete(self):
        fasta = self.fasta(UNIPROT_FASTA)
        source = self.h5_source({"P00001": rows(5, dtype=np.float16), "P00002": rows(6, dtype=np.float16),
                                 "P99999": rows(3)})
        result = self.run_import(fasta, source, label="uniprot_prot_t5", mode="float32")

        self.assertTrue(result["complete"])
        self.assertEqual(Path(result["output_hdf5"]), self.embed_dir / "set_[uniprot_prot_t5]_embeddings.h5")
        self.assertIsNone(result["report_path"])
        self.assertEqual((result["imported_now"], result["unused_source_ids"]), (2, 1))
        self.assertEqual(result["matched_by"], {"uniprot_accession": 2})
        manifest, arrays, attrs = self.read(result["output_hdf5"])
        self.assertEqual(manifest.model_name, "uniprot_prot_t5")
        self.assertEqual(manifest.saving_mode, "float32")
        self.assertEqual(manifest.feature_dimension, 4)
        header = manifest.headers[0]
        self.assertTrue(header.startswith("sp|P00001|AAA_ECOLI_Alpha_protein"))
        # float16 values are stored unchanged at float32.
        np.testing.assert_array_equal(arrays[header], rows(5, dtype=np.float16).astype(np.float32))
        self.assertEqual(arrays[header].dtype, np.float32)
        self.assertEqual(attrs["import_source"], "source.h5")
        self.assertEqual(attrs["import_reader"], "hdf5")
        self.assertEqual(attrs["import_source_dtype"], "float16")
        self.assertEqual(json.loads(attrs["import_matched_by"]), {"uniprot_accession": 2})

    def test_each_rule_finds_its_kind_of_id(self):
        fasta = self.fasta(">whole\nAAAA\n>first word\nCCCC\n>a/b c\nDDDD\n>tr|Q00003|N\nEEEE\n")
        source = self.h5_source({"whole": rows(4, seed=1), "first": rows(4, seed=2),
                                 "a_b_c": rows(4, seed=3), "Q00003": rows(4, seed=4)})
        result = self.run_import(fasta, source)
        self.assertTrue(result["complete"])
        self.assertEqual(result["matched_by"], {"header": 1, "first_word": 1,
                                                "sanitized_header": 1, "uniprot_accession": 1})

    def test_the_original_header_wins_over_its_first_word(self):
        fasta = self.fasta(">P1 extra words\nAAAA\n")
        source = self.h5_source({"P1 extra words": rows(4, seed=1), "P1": rows(4, seed=2)})
        result = self.run_import(fasta, source)
        _, arrays, _ = self.read(result["output_hdf5"])
        np.testing.assert_array_equal(next(iter(arrays.values())), rows(4, seed=1))

    def test_records_merged_by_sanitizing_try_every_original_header(self):
        # The sanitizer keeps the longest header of identical sequences; the source has the other.
        fasta = self.fasta(">short\nMKTAY\n>a_much_longer_header\nMKTAY\n")
        source = self.h5_source({"short": rows(5)})
        result = self.run_import(fasta, source)
        self.assertTrue(result["complete"])
        manifest, _, _ = self.read(result["output_hdf5"])
        self.assertEqual(manifest.headers, ["a_much_longer_header"])

    def test_an_id_two_records_could_take_is_not_used_until_a_mapping_file_says_which(self):
        fasta = self.fasta(">protein one\nAAAA\n>protein two\nCCCC\n")
        source = self.h5_source({"protein": rows(4), "k1": rows(4, seed=1), "k2": rows(4, seed=2)})
        result = self.run_import(fasta, source)
        self.assertFalse(result["complete"])
        self.assertEqual(result["ambiguous"], 2)
        report = self.report(result)
        self.assertEqual([row[1] for row in report], ["ambiguous", "ambiguous"])
        self.assertIn("ID mapping file", report[0][3])

        id_map = self.root / "map.tsv"
        id_map.write_text("# FASTA header\tsource ID\nprotein one\tk1\nprotein_two\tk2\n", encoding="utf-8")
        result = self.run_import(fasta, source, id_map=id_map)
        self.assertTrue(result["complete"])
        self.assertEqual(result["matched_by"], {"id_map": 2})
        self.assertFalse((self.embed_dir / "set_[my_model]_import_report.tsv").exists())

    def test_a_mapping_to_an_absent_id_is_reported_and_not_replaced_by_a_rule(self):
        fasta = self.fasta(">P1\nAAAA\n")
        source = self.h5_source({"P1": rows(4)})
        id_map = self.root / "map.tsv"
        id_map.write_text("P1\tnot_there\n", encoding="utf-8")
        result = self.run_import(fasta, source, id_map=id_map)
        self.assertFalse(result["complete"])
        self.assertEqual(self.report(result)[0][:3], ["P1", "unmapped", "not_there"])

    def test_a_mapping_file_that_cannot_be_read_stops_before_any_output(self):
        fasta = self.fasta(">P1\nAAAA\n")
        source = self.h5_source({"P1": rows(4)})
        for text, message in (("P1\n", "two tab-separated"), ("P1\ta\nP1\tb\n", "earlier line")):
            with self.subTest(text=text):
                id_map = self.root / "bad.tsv"
                id_map.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.run_import(fasta, source, id_map=id_map)
                self.assertFalse(self.embed_dir.exists())


class IdMapFileTests(ImportCase):
    """The mapping file: tab-separated, or comma-separated when it is a .csv file."""

    def read_map(self, name, text, encoding="utf-8"):
        path = self.root / name
        path.write_bytes(text.encode(encoding))
        return Embedding_Import.read_id_map(str(path))

    def test_a_csv_file_is_comma_separated_with_the_csv_quoting(self):
        mapping = self.read_map(
            "map.csv",
            '# FASTA header,source ID\r\nprotein one,k1\r\n"protein, two",k2\r\n"say ""hi""",k3,ignored\r\n\r\n',
        )
        self.assertEqual(mapping, {"protein one": "k1", "protein, two": "k2", 'say "hi"': "k3"})
        self.assertEqual(self.read_map("UPPER.CSV", "a,k1\n"), {"a": "k1"})

    def test_a_csv_file_maps_records_whose_headers_hold_a_comma(self):
        fasta = self.fasta(">protein one\nAAAA\n>protein, two\nCCCC\n")
        source = self.h5_source({"k1": rows(4, seed=1), "k2": rows(4, seed=2)})
        id_map = self.root / "map.csv"
        id_map.write_text('protein one,k1\n"protein, two",k2\n', encoding="utf-8")
        result = self.run_import(fasta, source, id_map=id_map)
        self.assertTrue(result["complete"])
        self.assertEqual(result["matched_by"], {"id_map": 2})

    def test_tsv_txt_and_other_files_stay_tab_separated(self):
        for name in ("map.tsv", "map.txt", "map"):
            with self.subTest(name=name):
                self.assertEqual(self.read_map(name, "a,b\tk1\nc\tk2\n"), {"a,b": "k1", "c": "k2"})

    def test_a_byte_order_mark_is_not_part_of_the_first_id(self):
        for name, separator in (("map.csv", ","), ("map.tsv", "\t")):
            with self.subTest(name=name):
                self.assertEqual(self.read_map(name, f"﻿a{separator}k1\n"), {"a": "k1"})

    def test_a_line_without_two_columns_names_the_file_the_line_and_what_was_expected(self):
        for name, text, line, kind in (
            ("bad.csv", "a,k1\nb\n", 2, "comma-separated"),
            ("bad.csv", "a,k1\n\nb,\n", 3, "comma-separated"),
            ("bad.tsv", "a\tk1\nb,k2\n", 2, "tab-separated"),
        ):
            with self.subTest(name=name, text=text):
                with self.assertRaises(Embedding_Import.EmbeddingImportError) as caught:
                    self.read_map(name, text)
                message = str(caught.exception)
                for part in (name, f"Line {line} ", f"two {kind} columns", "a FASTA header and a source ID"):
                    self.assertIn(part, message)


class ArrayCheckTests(ImportCase):
    def refused(self, array, mode="float32", length=6):
        fasta = self.fasta(f">P1\n{'ACDEFGHIKL'[:length]}\n")
        source = self.h5_source({"P1": array})
        result = self.run_import(fasta, source, mode=mode)
        self.assertFalse(result["complete"])
        self.assertEqual(result["invalid"], 1)
        (row,) = self.report(result)
        self.assertEqual(row[:3], ["P1", "invalid", "P1"])
        return row[3]

    def test_a_per_protein_vector_is_refused_saying_alignment_needs_residues(self):
        reason = self.refused(np.ones(4, dtype=np.float32))
        self.assertIn("per-protein embedding", reason)
        self.assertIn("one vector per residue", reason)

    def test_rows_for_start_and_end_tokens_are_named_as_the_likely_cause(self):
        self.assertIn("start and end tokens", self.refused(rows(8)))
        self.assertIn("start or end token", self.refused(rows(7)))
        self.assertIn("has 3 rows for a sequence of 6 residues", self.refused(rows(3)))

    def test_values_that_are_not_finite_or_not_floating_point_are_refused(self):
        bad = rows(6)
        bad[2, 1] = np.nan
        self.assertIn("NaN", self.refused(bad))
        self.assertIn("not floating-point", self.refused(np.ones((6, 4), dtype=np.int32)))

    def test_values_beyond_float16_are_refused_at_float16_and_kept_at_float32(self):
        large = rows(6) * 1e6
        self.assertIn("beyond the range of float16", self.refused(large, mode="float16"))
        fasta = self.fasta(">P2\nACDEFG\n", name="other.fasta")
        result = self.run_import(fasta, self.h5_source({"P2": large}, name="large.h5"), mode="float32")
        self.assertTrue(result["complete"])

    def test_one_feature_width_holds_for_the_whole_file(self):
        fasta = self.fasta(">P1\nACDE\n>P2\nACDEF\n")
        source = self.h5_source({"P1": rows(4, width=4), "P2": rows(5, width=3)})
        result = self.run_import(fasta, source)
        self.assertFalse(result["complete"])
        (row,) = self.report(result)
        self.assertEqual(row[0], "P2")
        self.assertIn("feature dimension 3; expected 4", row[3])


class SourceTests(ImportCase):
    FASTA = ">P1\nACDE\n>P2\nACDEF\n"

    def test_a_npz_file_is_read_by_its_array_names(self):
        source = self.root / "source.npz"
        np.savez(source, P1=rows(4), P2=rows(5))
        result = self.run_import(self.fasta(self.FASTA), source)
        self.assertTrue(result["complete"])
        self.assertEqual(result["reader"], "npz")

    def test_a_folder_of_npy_files_is_read_by_file_name_and_a_single_npy_is_refused(self):
        folder = self.root / "npy"
        folder.mkdir()
        np.save(folder / "P1.npy", rows(4))
        np.save(folder / "P2.npy", rows(5))
        result = self.run_import(self.fasta(self.FASTA), folder)
        self.assertTrue(result["complete"])
        self.assertEqual(result["reader"], "npy_folder")
        with self.assertRaisesRegex(ValueError, "Choose the folder"):
            self.run_import(self.fasta(self.FASTA, "again.fasta"), folder / "P1.npy")

    def test_an_original_id_attribute_counts_as_the_datasets_id(self):
        # bio_embeddings names datasets by a checksum and keeps the ID in original_id.
        source = self.h5_source({"9f2c": rows(4), "41aa": rows(5)}, original_ids={"9f2c": "P1", "41aa": "P2"})
        self.assertTrue(self.run_import(self.fasta(self.FASTA), source)["complete"])

    def test_sources_that_are_not_per_protein_arrays_are_refused_before_any_output(self):
        fasta = self.fasta(self.FASTA)
        emapssn = self.root / "own_[m]_embeddings.h5"
        with h5py.File(emapssn, "w") as hf:
            hf.attrs["model_name"] = "m"
            hf.create_group("embeddings")
        grouped = self.root / "grouped.h5"
        with h5py.File(grouped, "w") as hf:
            hf.create_group("inner").create_dataset("P1", data=rows(4))
        (self.root / "notes.txt").write_text("x", encoding="utf-8")
        for source, message in ((emapssn, "already an EMAP-SSN embedding file"),
                                (grouped, "no datasets at its top level"),
                                (self.root / "notes.txt", "not a source this tool reads"),
                                (self.root / "absent.h5", "not found")):
            with self.subTest(source=source.name):
                with self.assertRaisesRegex(ValueError, message):
                    self.run_import(fasta, source)
        self.assertFalse(self.embed_dir.exists())


class OutputTests(ImportCase):
    def test_no_matching_id_at_all_is_refused_before_a_file_is_made(self):
        fasta = self.fasta(">alpha\nACDE\n")
        with self.assertRaisesRegex(ValueError, "found its ID in the source"):
            self.run_import(fasta, self.h5_source({"other": rows(4)}))
        self.assertFalse(self.embed_dir.exists())

    def test_an_incomplete_file_is_refused_by_readers_and_a_later_run_finishes_it(self):
        fasta = self.fasta(">P1\nACDE\n>P2\nACDEF\n")
        result = self.run_import(fasta, self.h5_source({"P1": rows(4)}))
        self.assertFalse(result["complete"])
        self.assertEqual(self.report(result), [["P2", "missing", "", "no ID of this record is in the source"]])
        with h5py.File(result["output_hdf5"], "r") as hf:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_embedding_manifest(hf)

        result = self.run_import(fasta, self.h5_source({"P1": rows(4, seed=9), "P2": rows(5)}, name="more.h5"))
        self.assertTrue(result["complete"])
        self.assertEqual((result["imported_now"], result["already_present"]), (1, 1))
        _, arrays, attrs = self.read(result["output_hdf5"])
        np.testing.assert_array_equal(arrays["P1"], rows(4))  # kept from the first run
        self.assertEqual(attrs["import_source"], "more.h5")
        self.assertFalse(Path(self.embed_dir / "set_[my_model]_import_report.tsv").exists())

        # A run with nothing left to import leaves the record of where the arrays came from.
        result = self.run_import(fasta, self.h5_source({"P1": rows(4)}, name="third.h5"))
        self.assertEqual(result["imported_now"], 0)
        self.assertEqual(self.read(result["output_hdf5"])[2]["import_source"], "more.h5")

    def test_another_saving_mode_or_label_does_not_resume_a_file(self):
        fasta = self.fasta(">P1\nACDE\n")
        source = self.h5_source({"P1": rows(4)})
        output = self.run_import(fasta, source)["output_hdf5"]
        with self.assertRaisesRegex(ValueError, "float16"):
            self.run_import(fasta, source, mode="float16")
        os.rename(output, self.embed_dir / "set_[other]_embeddings.h5")
        with self.assertRaisesRegex(ValueError, "model 'my_model'"):
            self.run_import(fasta, source, label="other")

    def test_labels_of_shipped_models_and_unsafe_labels_are_refused(self):
        fasta = self.fasta(">P1\nACDE\n")
        source = self.h5_source({"P1": rows(4)})
        with self.assertRaisesRegex(ValueError, "imported_prot_t5"):
            self.run_import(fasta, source, label="prot_t5")
        for label in ("", "Upper", "a[b]", "a/b", "_lead", "x" * 65):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "Model label"):
                    self.run_import(fasta, source, label=label)
        self.assertFalse(self.embed_dir.exists())


class MainTests(ImportCase):
    def run_main(self, **settings):
        values = {"INPUT_FASTA": None, "SOURCE_PATH": None, "ID_MAP_TSV": None,
                  "MODEL_LABEL": None, "SAVING_MODE": "float32",
                  "FASTA_DIR": str(self.root), "EMBED_DIR": str(self.embed_dir), **settings}
        with mock.patch.multiple(Embedding_Import, **values), \
                mock.patch.object(Embedding_Import, "load_tool_settings", return_value=None), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = Embedding_Import.main([])
        return code, out.getvalue()

    def test_a_relative_source_is_read_in_the_fasta_folder_and_the_run_succeeds(self):
        self.fasta(UNIPROT_FASTA)
        self.h5_source({"P00001": rows(5), "P00002": rows(6)})
        code, out = self.run_main(INPUT_FASTA="set.fasta", SOURCE_PATH="source.h5", MODEL_LABEL="uniprot_prot_t5")
        self.assertEqual(code, 0, out)
        self.assertIn("Imported now: 2", out)
        self.assertTrue((self.embed_dir / "set_[uniprot_prot_t5]_embeddings.h5").exists())

    def test_an_incomplete_import_fails_the_run_and_names_the_report(self):
        self.fasta(UNIPROT_FASTA)
        self.h5_source({"P00001": rows(5)})
        code, out = self.run_main(INPUT_FASTA="set.fasta", SOURCE_PATH="source.h5", MODEL_LABEL="m")
        self.assertEqual(code, 1)
        self.assertIn("Missing from the source: 1", out)
        self.assertIn("set_[m]_import_report.tsv", out)
        self.assertIn("is incomplete", out)

    def test_missing_settings_and_shipped_labels_fail_with_a_message(self):
        self.assertEqual(self.run_main()[0], 1)
        self.fasta(UNIPROT_FASTA)
        code, out = self.run_main(INPUT_FASTA="set.fasta", SOURCE_PATH="source.h5", MODEL_LABEL="esm2_t6_8m")
        self.assertEqual(code, 1)
        self.assertIn("names a model EMAP-SSN computes itself", out)

    def test_a_missing_label_is_answered_as_before_and_an_empty_one_in_a_settings_document_is_missing_too(self):
        self.fasta(UNIPROT_FASTA)
        self.h5_source({"P00001": rows(5), "P00002": rows(6)})
        code, out = self.run_main(INPUT_FASTA="set.fasta", SOURCE_PATH="source.h5")
        self.assertEqual(code, 1)
        self.assertIn("MODEL_LABEL is not set.", out)
        self.assertFalse(self.embed_dir.exists())
        from tools.tool_helpers.Tool_Pipeline import apply_settings_document
        namespace = {"MODEL_LABEL": None, "SAVING_MODE": "float16"}
        apply_settings_document(namespace, {"Embedding_Import.py": {"MODEL_LABEL": "", "SAVING_MODE": "float32"}},
                                "Embedding_Import.py", str(self.root))
        self.assertEqual(namespace, {"MODEL_LABEL": None, "SAVING_MODE": "float32"})


def literal(script, name):
    """The literal a tool script assigns to name, as the Tools window reads its defaults."""
    tree = ast.parse((SRC / "tools" / script).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{script} assigns no {name}")


def generate_embeddings_default():
    return literal("Generate_Embeddings.py", "SAVING_MODE")


class SavingModeDefaultTests(unittest.TestCase):
    def test_the_default_follows_generate_embeddings_everywhere(self):
        from mcp_server.pipeline.Pipeline_Settings import _CONTRACTS
        default = generate_embeddings_default()
        self.assertEqual(literal("Embedding_Import.py", "SAVING_MODE"), default)
        for script in ("Generate_Embeddings.py", "Embedding_Import.py"):
            with self.subTest(script=script):
                prop = _CONTRACTS[script]["properties"]["SAVING_MODE"]
                self.assertEqual(prop["default"], default)
                self.assertEqual(prop["enum"], ["float32", "float16"])


class ToolsCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def window(self, sections=None):
        from tests.tools_gui_fixtures import isolated_tools_project
        root = isolated_tools_project(self, sections=sections)
        from PySide6.QtWidgets import QTextBrowser
        from EMAPSSN_Tools import ToolsGUI
        with mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser):
            window = ToolsGUI()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        path = next(p for p in window.script_data if Path(p).name == "Embedding_Import.py")
        return window, path, root

    def test_the_card_is_last_on_the_first_tab_and_offers_the_contracts_fields(self):
        import EMAPSSN_Tools
        from mcp_server.pipeline.Pipeline_Settings import _CONTRACTS
        window, path, _ = self.window()
        first_tab = list(window.MANUAL_SETTINGS["Sequence_and_Embedding_Preparation"]["scripts"])
        self.assertEqual(first_tab[-1], "Embedding_Import.py")
        self.assertEqual(EMAPSSN_Tools.TOOL_TITLES["Embedding_Import.py"], "Embedding Import")
        self.assertEqual(EMAPSSN_Tools.TOOL_ICONS["Embedding_Import.py"], "file-input")
        inputs = window.script_data[path]["inputs"]
        self.assertLessEqual(set(_CONTRACTS["Embedding_Import.py"]["properties"]), set(inputs))
        self.assertEqual(window._collect_tool_settings(path)["SAVING_MODE"], generate_embeddings_default())

    def test_the_source_row_picks_a_file_or_a_folder_and_the_mapping_row_a_file(self):
        window, path, _ = self.window()
        inputs = window.script_data[path]["inputs"]
        source, id_map = inputs["SOURCE_PATH"]["widget"], inputs["ID_MAP_TSV"]["widget"]
        self.assertEqual((source.browse_button.text(), source.folder_button.text()), ("File...", "Folder..."))
        self.assertIsNone(id_map.folder_button)
        folder = tempfile.gettempdir()
        with mock.patch("EMAPSSN_Tools.QFileDialog.getExistingDirectory", return_value=folder) as dialog:
            source.folder_button.click()
        self.assertEqual(dialog.call_args.args[1], "Select Embedding Source")
        self.assertEqual(source.line_edit.text(), os.path.normpath(folder))
        chosen = os.path.join(folder, "per-residue.h5")
        with mock.patch("EMAPSSN_Tools.QFileDialog.getOpenFileName", return_value=(chosen, "")) as dialog:
            source.browse_button.click()
        self.assertIn("*.npz", dialog.call_args.args[3])
        self.assertEqual(source.line_edit.text(), os.path.normpath(chosen))
        with mock.patch("EMAPSSN_Tools.QFileDialog.getOpenFileName", return_value=("", "")) as dialog:
            id_map.browse_button.click()
        self.assertEqual(dialog.call_args.args[1], "Select ID Mapping File")
        self.assertIn("*.tsv *.csv *.txt", dialog.call_args.args[3])

    def test_save_and_run_hands_both_paths_to_the_run_but_never_saves_them(self):
        window, path, root = self.window()
        inputs = window.script_data[path]["inputs"]
        source = str(Path(tempfile.gettempdir()) / "per-residue.h5")
        id_map = str(Path(tempfile.gettempdir()) / "map.tsv")
        inputs["SOURCE_PATH"]["widget"].line_edit.setText(source)
        inputs["ID_MAP_TSV"]["widget"].line_edit.setText(id_map)
        inputs["MODEL_LABEL"]["widget"].setText("uniprot_prot_t5")
        launched = []
        with mock.patch("EMAPSSN_Tools.launch_in_terminal", side_effect=lambda argv, **kw: launched.append(argv)):
            window.save_and_run(path)
        saved_text = (root / "tools_settings.json").read_text(encoding="utf-8")
        section = json.loads(saved_text)["Embedding_Import.py"]
        self.assertNotIn("SOURCE_PATH", section)
        self.assertNotIn("ID_MAP_TSV", section)
        self.assertEqual(section["MODEL_LABEL"], "uniprot_prot_t5")
        self.assertNotIn("per-residue.h5", saved_text)
        self.assertEqual(len(launched), 1)
        from tools.tool_helpers.Tool_Pipeline import forget_one_run_snapshot
        snapshot = Path(launched[0][-1])
        self.addCleanup(lambda: forget_one_run_snapshot(str(snapshot)))
        document = json.loads(snapshot.read_text(encoding="utf-8"))["Embedding_Import.py"]
        self.assertEqual((document["SOURCE_PATH"], document["ID_MAP_TSV"]),
                         (os.path.normpath(source), os.path.normpath(id_map)))

    def test_an_untouched_card_shows_an_empty_label_and_saves_no_text_none(self):
        window, path, root = self.window()
        label = window.script_data[path]["inputs"]["MODEL_LABEL"]["widget"]
        self.assertEqual(label.text(), "")
        self.assertEqual(window._collect_tool_settings(path)["MODEL_LABEL"], "")
        launched = []
        with mock.patch("EMAPSSN_Tools.launch_in_terminal", side_effect=lambda argv, **kw: launched.append(argv)):
            window.save_and_run(path)
        saved = json.loads((root / "tools_settings.json").read_text(encoding="utf-8"))["Embedding_Import.py"]
        self.assertEqual(saved.get("MODEL_LABEL", ""), "")
        self.assertNotIn("None", saved.values())
        self.assertEqual(len(launched), 1)
        from tools.tool_helpers.Tool_Pipeline import forget_one_run_snapshot
        snapshot = Path(launched[0][-1])
        self.addCleanup(lambda: forget_one_run_snapshot(str(snapshot)))
        document = json.loads(snapshot.read_text(encoding="utf-8"))["Embedding_Import.py"]
        self.assertNotIn("None", document.values())

    def test_every_text_field_whose_default_is_none_shows_empty_even_after_an_older_save(self):
        from PySide6.QtWidgets import QLineEdit
        saved = {"Embedding_Import.py": {"MODEL_LABEL": "None"},
                 "Embedding_PWA.py": {"REF_HEADER": "None", "TAR_HEADER": "None"},
                 "Embedding_SSEARCH.py": {"NORM_THRESHOLD": "None"}}
        for sections in ({}, saved):
            window, _, _ = self.window(sections=sections)
            for script, names in (("Embedding_Import.py", ("MODEL_LABEL",)),
                                  ("Embedding_PWA.py", ("REF_HEADER", "TAR_HEADER")),
                                  ("Embedding_SSEARCH.py", ("NORM_THRESHOLD",))):
                path = next(p for p in window.script_data if Path(p).name == script)
                for name in names:
                    with self.subTest(script=script, name=name, saved=bool(sections)):
                        field = window.script_data[path]["inputs"][name]["widget"]
                        self.assertIsInstance(field, QLineEdit)
                        self.assertEqual(field.text(), "")
                        self.assertEqual(window._collect_tool_settings(path)[name], "")

    # --- the card's rows ---

    def shown_card(self):
        from PySide6.QtWidgets import QLabel, QWidget
        window, path, _ = self.window()
        window.show()
        self.flush()
        inputs = window.script_data[path]["inputs"]
        field = inputs["MODEL_LABEL"]["widget"]
        tab = next(index for index in range(window.tabs.count()) if window.tabs.widget(index).isAncestorOf(field))
        window.tabs.setCurrentIndex(tab)
        self.flush()
        scroll = window.tabs.widget(tab)
        card = field.parentWidget()

        def label_x(text):
            labels = [label for label in card.findChildren(QLabel) if label.text() == text]
            self.assertEqual(len(labels), 1, text)
            return labels[0].mapTo(scroll.widget(), labels[0].rect().topLeft()).x()

        def box(widget):
            return widget.rect().translated(widget.mapTo(scroll.widget(), widget.rect().topLeft()))

        return window, path, scroll, card, label_x, box, QWidget

    def flush(self):
        for _ in range(4):
            self.app.processEvents()

    def resize_panel(self, window, width):
        right = window.splitter.widget(1).minimumSizeHint().width() + 4
        window.resize(width + right + 34, 850)
        window.splitter.setSizes([width + 4, right])
        self.flush()
        delta = width - window.tabs.currentWidget().width()
        left, right = window.splitter.sizes()
        window.splitter.setSizes([left + delta, right - delta])
        self.flush()

    def test_the_rows_hold_the_set_and_the_mapping_then_the_source_then_the_label_and_the_mode(self):
        window, path, scroll, card, label_x, box, _ = self.shown_card()
        self.resize_panel(window, 1400)
        inputs = window.script_data[path]["inputs"]
        fasta, id_map, source, label, mode = (box(inputs[name]["widget"]) for name in (
            "INPUT_FASTA", "ID_MAP_TSV", "SOURCE_PATH", "MODEL_LABEL", "SAVING_MODE"))
        self.assertLessEqual(abs(fasta.center().y() - id_map.center().y()), 3)
        self.assertLess(fasta.right(), id_map.left())
        self.assertLess(fasta.bottom(), source.top())
        self.assertLess(source.bottom(), label.top())
        self.assertLessEqual(abs(label.center().y() - mode.center().y()), 3)
        self.assertLess(label.right(), mode.left())
        self.assertEqual({fasta.left(), source.left(), label.left()}, {fasta.left()})
        self.assertGreater(source.width(), fasta.width())
        self.assertGreater(label_x("ID Mapping (.tsv/.csv):"), label_x("Sequence Set (.fasta):"))

    def test_the_saving_mode_label_starts_where_the_id_mapping_label_does_at_every_width(self):
        window, path, scroll, card, label_x, box, QWidget = self.shown_card()
        group = card.findChild(QWidget, "compactRow_MODEL_LABEL_SAVING_MODE")
        self.assertIsNotNone(group)
        for width in (None, 1400, 1900):
            if width:
                self.resize_panel(window, width)
            with self.subTest(width=width):
                if group.property("stacked"):
                    self.assertIsNone(width)
                    continue
                self.assertEqual(label_x("Saving Mode:"), label_x("ID Mapping (.tsv/.csv):"))
        self.resize_panel(window, 1400)
        self.assertFalse(group.property("stacked"))

    def test_the_tab_key_follows_the_new_rows(self):
        from PySide6.QtCore import Qt
        window, path, scroll, card, label_x, box, QWidget = self.shown_card()
        inputs = window.script_data[path]["inputs"]
        order = ("INPUT_FASTA", "ID_MAP_TSV", "SOURCE_PATH", "MODEL_LABEL", "SAVING_MODE")
        owner = {}
        for name in order:
            widget = inputs[name]["widget"]
            for member in [widget] + widget.findChildren(QWidget):
                owner[member] = name
        seen, current = [], inputs[order[0]]["widget"]
        for _ in range(800):
            name = owner.get(current)
            if name and current.focusPolicy() != Qt.FocusPolicy.NoFocus and name not in seen[-1:]:
                seen.append(name)
            current = current.nextInFocusChain()
            if len(seen) >= len(order):
                break
        self.assertEqual(seen, list(order))

    def test_the_rows_stack_without_overlap_when_the_panel_is_narrow(self):
        window, path, scroll, card, label_x, box, QWidget = self.shown_card()
        self.resize_panel(window, 600)
        for name in ("compactRow_INPUT_FASTA_ID_MAP_TSV", "compactRow_MODEL_LABEL_SAVING_MODE"):
            self.assertTrue(card.findChild(QWidget, name).property("stacked"), name)
        for widget in [card] + card.findChildren(QWidget):
            layout = widget.layout()
            if layout is None:
                continue
            children = [layout.itemAt(index).widget() for index in range(layout.count())]
            children = [child for child in children if child is not None and child.isVisibleTo(card)]
            for index, child in enumerate(children):
                self.assertTrue(widget.rect().contains(child.geometry()), child.objectName())
                for other in children[index + 1:]:
                    self.assertFalse(child.geometry().intersects(other.geometry()), type(child).__name__)


class McpValidationTests(unittest.TestCase):
    def validate(self, **parameters):
        from mcp_server.pipeline.Pipeline_Settings import normalize_pipeline_settings
        base = {"INPUT_FASTA": "set.fasta", "SOURCE_PATH": "/data/per-residue.h5", "MODEL_LABEL": "uniprot_prot_t5"}
        return normalize_pipeline_settings("embedding_import", str(ROOT), parameters={**base, **parameters},
                                           directories={"FASTA_DIR": "Input_Files", "EMBED_DIR": "Embeddings"})

    def test_a_new_label_is_valid_and_a_shipped_or_unsafe_one_is_not(self):
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["settings_document"]["Embedding_Import.py"]["SAVING_MODE"], generate_embeddings_default())
        for label, message in (("prot_t5", "computes itself"), ("Bad Label", "does not match")):
            with self.subTest(label=label):
                result = self.validate(MODEL_LABEL=label)
                self.assertFalse(result["valid"])
                self.assertEqual(result["errors"][0]["field"], "parameters.MODEL_LABEL")
                self.assertIn(message, result["errors"][0]["message"])

    def test_the_schema_example_is_itself_valid(self):
        from mcp_server.pipeline.Pipeline_Settings import get_pipeline_schema, normalize_pipeline_settings
        schema = get_pipeline_schema("embedding_import", str(ROOT))
        example = schema["example"]["parameters"]
        self.assertEqual(example["MODEL_LABEL"], "uniprot_prot_t5")
        result = normalize_pipeline_settings("embedding_import", str(ROOT), parameters=example,
                                             directories={"FASTA_DIR": "Input_Files", "EMBED_DIR": "Embeddings"})
        self.assertTrue(result["valid"], result["errors"])


if __name__ == "__main__":
    unittest.main()
