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
from mcp_server.Pipeline_Settings import (
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

    def test_conditional_requirements_and_combinations(self):
        cases = [
            ("embedding_msa", {"USE_SEQUENCE_FILTER": True}),
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
from mcp_server.Pipeline_Settings import get_pipeline_schema
from tools.tool_helpers.Tool_Pipeline import list_tool_specs
for spec in list_tool_specs():
    get_pipeline_schema(spec.tool_id, {str(ROOT)!r})
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
