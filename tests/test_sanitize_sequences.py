import ast
import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODULE_PATH = PROJECT_ROOT / "src" / "tools" / "Sanitize_Sequences.py"
SPEC = importlib.util.spec_from_file_location("sanitize_sequences", MODULE_PATH)
sanitize_sequences = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sanitize_sequences)

from utilities.Tool_Directories import (  # noqa: E402
    DEFAULT_DIRECTORY_PATHS,
    fill_missing_directory_defaults,
    project_directory_defaults,
)


EXPECTED_TOOL_DIRECTORIES = {
    "Align_Similarity_Matrix.py": {"EMBED_DIR", "NETWORK_DIR"},
    "Align_Substitution_Matrix.py": {"FASTA_DIR", "NETWORK_DIR"},
    "Embedding_Cropping.py": {"FASTA_DIR", "EMBED_DIR"},
    "Embedding_Extraction.py": {"FASTA_DIR", "EMBED_DIR"},
    "Embedding_Injection.py": {"FASTA_DIR", "EMBED_DIR"},
    "Embedding_MSA.py": {"FASTA_DIR", "EMBED_DIR", "NETWORK_DIR", "MSA_DIR"},
    "Embedding_PWA.py": {"EMBED_DIR", "REPORT_DIR"},
    "Embedding_SSEARCH.py": {"EMBED_DIR", "REPORT_DIR"},
    "Generate_Embeddings.py": {"FASTA_DIR", "EMBED_DIR"},
    "Network_Extraction.py": {"FASTA_DIR", "NETWORK_DIR", "PATH_DIR"},
    "Network_Injection.py": {"EMBED_DIR", "NETWORK_DIR"},
    "Parse_BLAST_Output.py": {"NETWORK_DIR"},
    "Sanitize_Sequences.py": {"FASTA_DIR"},
    "Sparse_MSA_Converter.py": {"MSA_DIR"},
}


class SanitizeSequencesTests(unittest.TestCase):
    def test_lossy_sequence_sanitization_remains_intentional(self):
        gap_sequence, _, _ = sanitize_sequences.sanitize_sequence("AC-D")
        punctuation_sequence, _, _ = sanitize_sequences.sanitize_sequence("AC?D")

        self.assertEqual(gap_sequence, "ACXD")
        self.assertEqual(punctuation_sequence, "ACXD")

    def test_header_filter_is_case_sensitive_and_none_is_literal(self):
        self.assertTrue(
            sanitize_sequences.should_remove_by_header(
                "protein with None assigned", "None"
            )
        )
        self.assertFalse(
            sanitize_sequences.should_remove_by_header(
                "protein with none assigned", "None"
            )
        )
        self.assertTrue(
            sanitize_sequences.should_remove_by_header(
                "protein with None assigned", None
            )
        )
        self.assertFalse(
            sanitize_sequences.should_remove_by_header("protein", "   ")
        )

    def test_header_sanitization_collapses_repeated_underscores(self):
        header, was_modified = sanitize_sequences.sanitize_header(
            "Alpha??__Beta"
        )

        self.assertEqual(header, "Alpha_Beta")
        self.assertTrue(was_modified)

    def test_configuration_requires_an_explicit_output_directory(self):
        with self.assertRaisesRegex(ValueError, "output directory"):
            sanitize_sequences.validate_configuration(
                None,
                "input.fasta",
                "output.fasta",
            )

    def test_preferred_header_reports_the_header_actually_duplicated(self):
        (
            best_header,
            discarded_headers,
            duplicate_headers,
            duplicate_count,
        ) = sanitize_sequences.select_preferred_header(
            ["duplicate", "duplicate", "a much longer header"]
        )

        self.assertEqual(best_header, "a much longer header")
        self.assertEqual(discarded_headers, ["duplicate"])
        self.assertEqual(duplicate_headers, {"duplicate"})
        self.assertEqual(duplicate_count, 1)

    def test_longest_header_wins_even_when_shorter_header_contains_sid(self):
        best_header, _, _, _ = sanitize_sequences.select_preferred_header(
            ["sid|x", "a much longer descriptive header"]
        )

        self.assertEqual(best_header, "a much longer descriptive header")

    def test_generated_suffixes_do_not_collide_with_existing_headers(self):
        header_to_seqs = {
            "A": ["AAA", "CCC"],
            "A_1": ["GGG"],
        }

        assigned = sanitize_sequences.allocate_unique_headers(header_to_seqs)
        flattened = [
            header
            for base_header in header_to_seqs
            for header in assigned[base_header]
        ]

        self.assertEqual(assigned["A_1"], ["A_1"])
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertNotIn("A_1", assigned["A"])

    def test_empty_destructive_overwrite_is_refused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = pathlib.Path(temp_dir) / "input.fasta"
            output_path.write_text(">original\nAAAA\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "zero output sequences"):
                sanitize_sequences.write_fasta_atomic(
                    output_path,
                    [],
                    [],
                    refuse_empty=True,
                )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                ">original\nAAAA\n",
            )

    def test_failed_atomic_replace_preserves_existing_file_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = pathlib.Path(temp_dir) / "input.fasta"
            output_path.write_text(">original\nAAAA\n", encoding="utf-8")

            with mock.patch.object(
                sanitize_sequences.os,
                "replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated replace failure"):
                    sanitize_sequences.write_fasta_atomic(
                        output_path,
                        ["replacement"],
                        ["CCCC"],
                        refuse_empty=True,
                    )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                ">original\nAAAA\n",
            )
            self.assertEqual(list(pathlib.Path(temp_dir).glob("*.tmp")), [])


class ToolDirectoryDefaultTests(unittest.TestCase):
    def test_alignment_reports_default_to_analysis_results(self):
        self.assertEqual(
            DEFAULT_DIRECTORY_PATHS["REPORT_DIR"],
            os.path.join("Analysis_Results", "Alignment_Report"),
        )

    def test_empty_settings_receive_all_gui_defaults(self):
        settings = {}

        result = fill_missing_directory_defaults(settings)

        self.assertIs(result, settings)
        self.assertEqual(settings["DIRECTORIES"], DEFAULT_DIRECTORY_PATHS)

    def test_blank_values_are_filled_without_replacing_custom_paths(self):
        settings = {
            "DIRECTORIES": {
                "FASTA_DIR": "/custom/fasta",
                "MSA_DIR": "  ",
                "BLASTP_DIR": "/custom/blast/bin",
            }
        }

        fill_missing_directory_defaults(settings)

        self.assertEqual(settings["DIRECTORIES"]["FASTA_DIR"], "/custom/fasta")
        self.assertEqual(
            settings["DIRECTORIES"]["MSA_DIR"],
            DEFAULT_DIRECTORY_PATHS["MSA_DIR"],
        )
        self.assertEqual(
            settings["DIRECTORIES"]["BLASTP_DIR"], "/custom/blast/bin"
        )

    def test_project_defaults_are_absolute_and_project_anchored(self):
        defaults = project_directory_defaults(PROJECT_ROOT)

        for key, relative_path in DEFAULT_DIRECTORY_PATHS.items():
            self.assertTrue(os.path.isabs(defaults[key]))
            self.assertEqual(
                defaults[key],
                os.path.normpath(os.path.join(PROJECT_ROOT, relative_path)),
            )

    def test_save_and_run_populates_missing_global_directories(self):
        source = (PROJECT_ROOT / "src" / "SSN_Tools.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        save_and_run = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "save_and_run"
        )

        calls = {
            node.func.id
            for node in ast.walk(save_and_run)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("fill_missing_directory_defaults", calls)

    def test_every_tool_uses_the_shared_project_anchored_fallbacks(self):
        tools_dir = PROJECT_ROOT / "src" / "tools"
        for filename, expected_directories in EXPECTED_TOOL_DIRECTORIES.items():
            with self.subTest(tool=filename):
                source = (tools_dir / filename).read_text(encoding="utf-8")
                tree = ast.parse(source)
                assignments = {}

                for node in tree.body:
                    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                        continue
                    target = node.targets[0]
                    if not isinstance(target, ast.Name):
                        continue
                    value = node.value
                    if (
                        isinstance(value, ast.Subscript)
                        and isinstance(value.value, ast.Name)
                        and value.value.id == "_DEFAULT_DIRECTORIES"
                        and isinstance(value.slice, ast.Constant)
                    ):
                        assignments[target.id] = value.slice.value

                for directory_name in expected_directories:
                    self.assertEqual(assignments.get(directory_name), directory_name)


if __name__ == "__main__":
    unittest.main()
