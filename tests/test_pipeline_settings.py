import ast
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jsonschema import Draft202012Validator
from mcp_server.pipeline.Pipeline_Settings import (
    PipelineSettingsError, get_pipeline_schema, normalize_pipeline_settings,
    serialize_export_settings,
)
from tools.tool_helpers.Tool_Pipeline import list_tool_specs


class PipelineSettingsTests(unittest.TestCase):
    def preview(self, tool="sanitize_sequences", **kwargs):
        return normalize_pipeline_settings(tool, ROOT, **kwargs)

    def test_all_contracts_examples_defaults_and_gui_names(self):
        gui_tree = ast.parse((ROOT / "src/EMAPSSN_Tools.py").read_text(encoding="utf-8"))
        for spec in list_tool_specs():
            with self.subTest(tool=spec.tool_id):
                discovery = get_pipeline_schema(spec.tool_id, ROOT)
                schema = discovery["parameters_schema"]
                Draft202012Validator.check_schema(schema)
                example = discovery["example"]["parameters"]
                result = self.preview(spec.tool_id, parameters=example)
                self.assertTrue(result["valid"], result["errors"])
                source = ast.parse((ROOT / spec.relative_script).read_text(encoding="utf-8"))
                defaults = {}
                for node in source.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                try: defaults[target.id] = ast.literal_eval(node.value)
                                except (ValueError, TypeError): pass
                for key, field in schema["properties"].items():
                    self.assertEqual(field["default"], defaults[key], key)
                    self.assertTrue(field["description"])
                exported_keys = set()
                for node in ast.walk(gui_tree):
                    if not isinstance(node, ast.Dict): continue
                    for key, value in zip(node.keys, node.values):
                        if not isinstance(key, ast.Constant) or key.value != spec.script_name or not isinstance(value, ast.List): continue
                        for field in value.elts:
                            if not isinstance(field, ast.Dict): continue
                            for k, v in zip(field.keys, field.values):
                                if isinstance(k, ast.Constant) and k.value == "var_name" and isinstance(v, ast.Constant) and v.value in defaults:
                                    exported_keys.add(v.value)
                self.assertEqual(exported_keys, set(schema["properties"]))
                # A complete GUI configuration round-trips with native values.
                values = deepcopy(result["settings_document"][spec.settings_section])
                for key in ("BATCH_SIZE", "NORM_THRESHOLD", "HOST_CACHE_GB"):
                    if key in values: values[key] = str(values[key])
                exported = serialize_export_settings(spec.tool_id, values, ROOT)
                self.assertTrue(self.preview(spec.tool_id, parameters=exported)["valid"])

    def test_strict_types_keys_ranges_and_finite_values(self):
        cases = [
            ("sanitize_sequences", {"OVER_WRTE": True}, "OVER_WRTE"),
            ("sanitize_sequences", {"OVER_WRITE": "false"}, "OVER_WRITE"),
            ("sanitize_sequences", {"MIN_SEQ_LENGTH": "10"}, "MIN_SEQ_LENGTH"),
            ("sanitize_sequences", {"MIN_SEQ_LENGTH": True}, "MIN_SEQ_LENGTH"),
            ("sanitize_sequences", {"MIN_SEQ_LENGTH": -1}, "MIN_SEQ_LENGTH"),
            ("sanitize_sequences", {"INPUT_FASTA": []}, "INPUT_FASTA"),
            ("sanitize_sequences", {"INPUT_FASTA": " "}, "INPUT_FASTA"),
            ("embedding_ssearch", {"NORM_THRESHOLD": float("nan")}, "NORM_THRESHOLD"),
            ("embedding_ssearch", {"NORM_THRESHOLD": float("inf")}, "NORM_THRESHOLD"),
            ("embedding_ssearch", {"ALIGNMENT_MODE": "bogus"}, "ALIGNMENT_MODE"),
            ("embedding_ssearch", {"DEVICE_SELECTION": "gpu"}, "DEVICE_SELECTION"),
            ("align_similarity_matrix", {"HOST_CACHE_GB": "4"}, "HOST_CACHE_GB"),
        ]
        for tool, overrides, field in cases:
            with self.subTest(tool=tool, overrides=overrides):
                values = get_pipeline_schema(tool, ROOT)["example"]["parameters"]
                values.update(overrides)
                result = self.preview(tool, parameters=values)
                self.assertFalse(result["valid"])
                self.assertTrue(any(e["field"].endswith(field) for e in result["errors"]), result)

    def test_input_forms_and_full_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            values = {"INPUT_FASTA": "sample.fasta"}
            dirs = {"FASTA_DIR": "Sequences"}
            doc = {"DIRECTORIES": {**dirs, "EMBED_DIR": "Other"}, "Sanitize_Sequences.py": values,
                   "Embedding_MSA.py": {"unrelated": "left alone"}}
            path = Path(temp) / "export.json"
            path.write_text(json.dumps(doc))
            results = [self.preview(parameters=values, directories=dirs),
                       self.preview(settings_document=doc), self.preview(settings_path=str(path))]
            self.assertTrue(all(r["valid"] for r in results))
            self.assertEqual(results[0]["settings_document"], results[1]["settings_document"])
            self.assertEqual(results[0]["settings_document"], results[2]["settings_document"])
            for kwargs in ({}, {"parameters": {}, "settings_document": doc},
                           {"settings_path": str(path), "directories": {}},
                           {"settings_document": {**doc, "Typo.py": {}}},
                           {"parameters": values, "directories": {"FAST_DIR": "x"}},
                           {"settings_document": {"Sanitize_Sequences.py": values}},
                           {"parameters": values, "directories": {"FASTA_DIR": 1}}):
                self.assertFalse(self.preview(**kwargs)["valid"], kwargs)
            # String coercion is forbidden for old exports too.
            doc["Sanitize_Sequences.py"]["OVER_WRITE"] = "true"
            self.assertFalse(self.preview(settings_document=doc)["valid"])
            path.write_text(json.dumps(doc))
            self.assertFalse(self.preview(settings_path=str(path))["valid"])

    def test_parameters_mode_falls_back_to_saved_tool_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            custom_msa_dir = root / "OneDrive" / "Multiple_Alignments"
            (root / "tools_settings.json").write_text(json.dumps(
                {"DIRECTORIES": {"MSA_DIR": str(custom_msa_dir)}}
            ))
            result = normalize_pipeline_settings(
                "sparse_msa_converter", root, parameters={"CONVERT_ALL": True},
            )
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(result["effective_directories"]["MSA_DIR"], str(custom_msa_dir))
            self.assertFalse((root / "Input_Files").exists())

            (root / "tools_settings.json").unlink()
            result = normalize_pipeline_settings(
                "sparse_msa_converter", root, parameters={"CONVERT_ALL": True},
            )
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(
                result["effective_directories"]["MSA_DIR"],
                str(root / "Input_Files" / "Multiple_Alignments"),
            )

    def test_an_unreadable_saved_file_fails_instead_of_using_default_directories(self):
        # Such a file used to count as no saved directories, so jobs ran on the defaults.
        unreadable = {
            "corrupt": ('{"DIRECTORIES": {"MSA_DIR": "custom"},}', r"line 1 column \d+"),
            "empty": ("", r"line 1 column 1 "),
            "not an object": ('[{"MSA_DIR": "custom"}]', "must contain a JSON object"),
        }
        for label, (content, reason) in unreadable.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                settings = root / "tools_settings.json"
                settings.write_text(content, encoding="utf-8")

                result = normalize_pipeline_settings(
                    "sparse_msa_converter", root, parameters={"CONVERT_ALL": "yes"},
                )
                self.assertFalse(result["valid"])
                self.assertIsNone(result["settings_document"])
                errors = {error["field"]: error["message"] for error in result["errors"]}
                self.assertIn(str(settings), errors["directories"])
                self.assertRegex(errors["directories"], reason)
                # Other problems are still reported with it.
                self.assertIn("parameters.CONVERT_ALL", errors)

                # The file is not needed when every directory is given.
                given = normalize_pipeline_settings(
                    "sparse_msa_converter", root, parameters={"CONVERT_ALL": True},
                    directories={"MSA_DIR": str(root / "msa")},
                )
                self.assertTrue(given["valid"], given["errors"])
                self.assertEqual(settings.read_text(encoding="utf-8"), content)
                self.assertEqual(os.listdir(root), ["tools_settings.json"])

    def test_conditional_requirements_and_combinations(self):
        cases = [
            ("embedding_msa", {"USE_SEQUENCE_FILTER": True}),
            ("embedding_msa", {"ALIGNMENT_SCORE": "local"}),
            ("embedding_pwa", {"MANUAL_REF_SEQ": False}),
            ("embedding_pwa", {"REF_SEQUENCE": "---"}),
            ("embedding_ssearch", {"MANUAL_QUERY_SEQ": True, "QUERY_SEQUENCE": ""}),
            ("sparse_msa_converter", {"CONVERT_ALL": False}),
            ("sanitize_sequences", {"ENABLE_LENGTH_FILTER": True, "MIN_SEQ_LENGTH": 10, "MAX_SEQ_LENGTH": 2}),
            ("parse_blast_output", {"BLAST_LAYOUT": "custom_columns", "EVALUE_COLUMN": 1}),
            ("align_similarity_matrix", {"EXECUTION_MODE": "tiled", "DEVICE_SELECTION": "cpu"}),
            ("embedding_ssearch", {"ACCELERATOR_PRECISION": "tf32", "DEVICE_SELECTION": "mps"}),
        ]
        for tool, overrides in cases:
            with self.subTest(tool=tool, overrides=overrides):
                values = get_pipeline_schema(tool, ROOT)["example"]["parameters"]
                values.update(overrides)
                self.assertFalse(self.preview(tool, parameters=values)["valid"])
        self.assertTrue(self.preview(parameters={"INPUT_FASTA": "x", "ENABLE_LENGTH_FILTER": True,
                                                "MIN_SEQ_LENGTH": 10, "MAX_SEQ_LENGTH": 0})["valid"])
        # PWA deliberately defaults blank stored headers to database entries.
        self.assertTrue(self.preview("embedding_pwa", parameters={"INPUT_EMBED": "x.h5"})["valid"])
        # Local MSA scores need a sequence-length normalization; the default
        # alignment_length is reported against NORMALIZATION_MODE.
        msa = get_pipeline_schema("embedding_msa", ROOT)["example"]["parameters"]
        rejected = self.preview("embedding_msa", parameters={**msa, "ALIGNMENT_SCORE": "local"})
        self.assertEqual([e["field"] for e in rejected["errors"]], ["parameters.NORMALIZATION_MODE"])
        self.assertTrue(self.preview("embedding_msa", parameters={
            **msa, "ALIGNMENT_SCORE": "local", "NORMALIZATION_MODE": "longer_sequence"})["valid"])
        # The same rule applies to SSEARCH, whose defaults are local + longer_sequence.
        search = get_pipeline_schema("embedding_ssearch", ROOT)["example"]["parameters"]
        rejected = self.preview("embedding_ssearch", parameters={**search, "NORM_MODE": "alignment_length"})
        self.assertEqual([e["field"] for e in rejected["errors"]], ["parameters.NORM_MODE"])
        self.assertTrue(self.preview("embedding_ssearch", parameters={
            **search, "ALIGNMENT_MODE": "global", "NORM_MODE": "alignment_length"})["valid"])

    def test_score_metric_and_bit_score_column_are_settings_of_both_blast_tools(self):
        for tool, script in (("align_substitution_matrix", "Align_Substitution_Matrix.py"),
                             ("parse_blast_output", "Parse_BLAST_Output.py")):
            with self.subTest(tool=tool):
                schema = get_pipeline_schema(tool, ROOT)["parameters_schema"]["properties"]
                self.assertEqual(schema["SCORE_METRIC"]["enum"], ["evalue", "efi_alignment_score"])
                self.assertEqual(schema["SCORE_METRIC"]["default"], "evalue")
                values = get_pipeline_schema(tool, ROOT)["example"]["parameters"]
                for metric in ("evalue", "efi_alignment_score"):
                    result = self.preview(tool, parameters={**values, "SCORE_METRIC": metric})
                    self.assertTrue(result["valid"], result["errors"])
                    self.assertEqual(result["settings_document"][script]["SCORE_METRIC"], metric)
                rejected = self.preview(tool, parameters={**values, "SCORE_METRIC": "alignment_score"})
                self.assertFalse(rejected["valid"])
                self.assertEqual([e["field"] for e in rejected["errors"]], ["parameters.SCORE_METRIC"])
        self.assertNotIn("BITSCORE_COLUMN", get_pipeline_schema("align_substitution_matrix", ROOT)["parameters_schema"]["properties"])
        parse = get_pipeline_schema("parse_blast_output", ROOT)
        self.assertEqual(parse["parameters_schema"]["properties"]["BITSCORE_COLUMN"]["default"], 12)
        values = parse["example"]["parameters"]
        for bad in (0, "12"):
            rejected = self.preview("parse_blast_output", parameters={**values, "BITSCORE_COLUMN": bad})
            self.assertEqual([e["field"] for e in rejected["errors"]], ["parameters.BITSCORE_COLUMN"])

    def test_similarity_matrix_is_a_pairwise_alignment_setting_with_the_blast_choices(self):
        discovery = get_pipeline_schema("embedding_pwa", ROOT)
        schema = discovery["parameters_schema"]["properties"]["SIMILARITY_MATRIX"]
        blast = get_pipeline_schema("align_substitution_matrix", ROOT)["parameters_schema"]["properties"]["MATRIX"]
        self.assertEqual(schema["default"], "BLOSUM62")
        self.assertEqual(schema["enum"], blast["enum"])
        self.assertEqual(schema["enum"], ["BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80", "BLOSUM90",
                                          "PAM30", "PAM70", "PAM250"])
        self.assertIn("similar", schema["description"])
        self.assertIn("similarity", " ".join(discovery["notes"]))
        values = discovery["example"]["parameters"]
        for matrix in schema["enum"]:
            result = self.preview("embedding_pwa", parameters={**values, "SIMILARITY_MATRIX": matrix})
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(result["settings_document"]["Embedding_PWA.py"]["SIMILARITY_MATRIX"], matrix)
        for bad in ("blosum62", "BLOSUM100", "", None, 62):
            rejected = self.preview("embedding_pwa", parameters={**values, "SIMILARITY_MATRIX": bad})
            self.assertFalse(rejected["valid"], bad)
            self.assertEqual({e["field"] for e in rejected["errors"]}, {"parameters.SIMILARITY_MATRIX"})

    def test_similarity_matrix_is_a_database_search_setting_with_the_same_choices(self):
        from utilities.Similarity_Matrices import SIMILARITY_MATRICES

        discovery = get_pipeline_schema("embedding_ssearch", ROOT)
        schema = discovery["parameters_schema"]["properties"]["SIMILARITY_MATRIX"]
        pairwise = get_pipeline_schema("embedding_pwa", ROOT)["parameters_schema"]["properties"]["SIMILARITY_MATRIX"]
        blast = get_pipeline_schema("align_substitution_matrix", ROOT)["parameters_schema"]["properties"]["MATRIX"]
        # One list for the BLAST tool and both tools that count similar pairs, the shared module's.
        self.assertEqual(schema["enum"], list(SIMILARITY_MATRICES))
        self.assertEqual(pairwise["enum"], list(SIMILARITY_MATRICES))
        self.assertEqual(blast["enum"], list(SIMILARITY_MATRICES))
        self.assertEqual(schema, pairwise)
        self.assertEqual(schema["default"], "BLOSUM62")
        self.assertIn("similar", schema["description"])
        self.assertIn("similarity", " ".join(discovery["notes"]))
        self.assertIn("Percent_Similarity", " ".join(discovery["notes"]))
        values = discovery["example"]["parameters"]
        omitted = self.preview("embedding_ssearch", parameters=values)
        self.assertTrue(omitted["valid"], omitted["errors"])
        self.assertEqual(omitted["settings_document"]["Embedding_SSEARCH.py"]["SIMILARITY_MATRIX"], "BLOSUM62")
        for matrix in schema["enum"]:
            result = self.preview("embedding_ssearch", parameters={**values, "SIMILARITY_MATRIX": matrix})
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(result["settings_document"]["Embedding_SSEARCH.py"]["SIMILARITY_MATRIX"], matrix)
        for bad in ("blosum62", "BLOSUM100", "", None, 62):
            rejected = self.preview("embedding_ssearch", parameters={**values, "SIMILARITY_MATRIX": bad})
            self.assertFalse(rejected["valid"], bad)
            self.assertEqual({e["field"] for e in rejected["errors"]}, {"parameters.SIMILARITY_MATRIX"})

    def test_custom_columns_are_distinct_for_the_score_that_is_read(self):
        values = get_pipeline_schema("parse_blast_output", ROOT)["example"]["parameters"]
        custom = {**values, "BLAST_LAYOUT": "custom_columns", "QUERY_COLUMN": 1, "SUBJECT_COLUMN": 2}
        # E-value scores read EVALUE_COLUMN only, EFI alignment scores BITSCORE_COLUMN only.
        for metric, column, other, message in (
            ("evalue", "EVALUE_COLUMN", "BITSCORE_COLUMN", "Query, subject, and E-value columns must be distinct."),
            ("efi_alignment_score", "BITSCORE_COLUMN", "EVALUE_COLUMN",
             "Query, subject, and bit score columns must be distinct."),
        ):
            with self.subTest(metric=metric):
                clash = self.preview("parse_blast_output", parameters={**custom, "SCORE_METRIC": metric, column: 2})
                self.assertFalse(clash["valid"])
                self.assertEqual(clash["errors"], [{"field": f"parameters.{column}", "message": message}])
                # The column that is not read may repeat another one.
                self.assertTrue(self.preview("parse_blast_output", parameters={
                    **custom, "SCORE_METRIC": metric, column: 3, other: 1})["valid"])

    def test_exports_preserve_strings_and_plots_and_allow_incomplete_inputs(self):
        values = {"BATCH_SIZE": "500000", "HOST_CACHE_GB": "2.5"}
        self.assertEqual(serialize_export_settings("align_similarity_matrix", values, ROOT),
                         {"BATCH_SIZE": 500000, "HOST_CACHE_GB": 2.5})
        self.assertEqual(values["BATCH_SIZE"], "500000")
        for raw in ("", "None", "null"):
            self.assertIsNone(serialize_export_settings("embedding_ssearch", {"NORM_THRESHOLD": raw}, ROOT)["NORM_THRESHOLD"])
        text = {"QUERY_HEADER": "None", "OUTPUT_NAME": "%NAME%", "QUERY_SEQUENCE": "AC D"}
        self.assertEqual(serialize_export_settings("embedding_ssearch", text, ROOT), text)
        self.assertEqual(serialize_export_settings("embedding_pwa", {"HIGHLIGHT_POSITIONS": "1,3-5"}, ROOT),
                         {"HIGHLIGHT_POSITIONS": "1,3-5"})
        for values in ({"BATCH_SIZE": "oops"}, {"BACH_SIZE": 10}):
            with self.assertRaises(PipelineSettingsError): serialize_export_settings("align_similarity_matrix", values, ROOT)
        exported = serialize_export_settings("generate_embeddings", {"INPUT_FASTA": "", "MODEL_NAME": ""}, ROOT)
        self.assertFalse(self.preview("generate_embeddings", parameters=exported)["valid"])
        msa = {"INPUT_EMBED": "x", "INPUT_NETWORK": "y", "SHOW_REGRESSION_PLOT": True}
        exported = serialize_export_settings("embedding_msa", msa, ROOT)
        self.assertTrue(exported["SHOW_REGRESSION_PLOT"])
        result = self.preview("embedding_msa", parameters=exported)
        self.assertFalse(result["settings_document"]["Embedding_MSA.py"]["SHOW_REGRESSION_PLOT"])
        self.assertEqual(len(result["overrides"]), 1)
        msa["SHOW_REGRESSION_PLOT"] = "true"
        self.assertFalse(self.preview("embedding_msa", parameters=msa)["valid"])

    def test_the_length_histogram_switch_defaults_on_and_is_kept_headless(self):
        discovery = get_pipeline_schema("sanitize_sequences", ROOT)
        field = discovery["parameters_schema"]["properties"]["SHOW_LENGTH_HISTOGRAM"]
        self.assertEqual(field["type"], "boolean")
        self.assertIs(field["default"], True)
        self.assertEqual(field["description"], (
            "Show Length Histogram: Displays the length histogram of the sanitized sequences "
            "when the run ends.\nA run from a settings file (CLI or MCP) prints the same bins "
            "as a table instead. Turn off to skip both."))
        self.assertIn("SHOW_LENGTH_HISTOGRAM", discovery["headless_behavior"]["length_distribution"])
        result = self.preview(parameters={"INPUT_FASTA": "x"})
        self.assertIs(result["settings_document"]["Sanitize_Sequences.py"]["SHOW_LENGTH_HISTOGRAM"], True)
        self.assertIs(result["applied_defaults"]["parameters.SHOW_LENGTH_HISTOGRAM"], True)
        # Unlike SHOW_REGRESSION_PLOT, a headless job keeps the supplied value.
        result = self.preview(parameters={"INPUT_FASTA": "x", "SHOW_LENGTH_HISTOGRAM": False})
        self.assertIs(result["settings_document"]["Sanitize_Sequences.py"]["SHOW_LENGTH_HISTOGRAM"], False)
        self.assertEqual(result["overrides"], [])
        result = self.preview(parameters={"INPUT_FASTA": "x", "SHOW_LENGTH_HISTOGRAM": "false"})
        self.assertFalse(result["valid"])
        self.assertEqual([e["field"] for e in result["errors"]], ["parameters.SHOW_LENGTH_HISTOGRAM"])

    def test_preview_has_no_files_and_no_gui_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            shared = Path(temp) / "tools_settings.json"
            shared.write_text(json.dumps({"Sanitize_Sequences.py": {"OVER_WRITE": True}}))
            result = normalize_pipeline_settings("sanitize_sequences", temp, parameters={"INPUT_FASTA": "x"})
            self.assertTrue(result["valid"])
            self.assertFalse(result["settings_document"]["Sanitize_Sequences.py"]["OVER_WRITE"])
            self.assertEqual(list(Path(temp).iterdir()), [shared])
            self.assertEqual(result["effective_directories"]["FASTA_DIR"], str(Path(temp)/"Input_Files"/"Sequence_Sets"))
            result = normalize_pipeline_settings("align_substitution_matrix", temp,
                parameters={"INPUT_FASTA": "future.fasta", "BLASTP_DIR": "bin"})
            self.assertEqual(result["settings_document"]["Align_Substitution_Matrix.py"]["BLASTP_DIR"],
                             str(Path(temp) / "bin"))

    def test_discovery_does_not_import_computational_or_gui_modules(self):
        code = f"""
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
for name in ('torch', 'numpy', 'matplotlib', 'PySide6'):
    sys.modules[name] = None
from mcp_server.pipeline.Pipeline_Settings import get_pipeline_schema
from tools.tool_helpers.Tool_Pipeline import list_tool_specs
for spec in list_tool_specs():
    get_pipeline_schema(spec.tool_id, {str(ROOT)!r})
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
