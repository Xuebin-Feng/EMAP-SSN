"""Sanitize_Sequences tool: its own rules, output writing and command-line runs.

The header and sequence rules it re-imports from utilities.Sequence_Utils are
covered in test_fasta_sanitization.py.
"""
import contextlib
import io
import json
import os
import pathlib
import sys
import subprocess
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODULE_PATH = PROJECT_ROOT / "src" / "tools" / "Sanitize_Sequences.py"
# A package import, so the tests-package hook keeps <project>/tools_settings.json
# out of the module's import-time settings.
from tools import Sanitize_Sequences as sanitize_sequences  # noqa: E402


def _tool_environment(temp_dir):
    """Subprocess environment whose import-time settings read finds no file.

    A tool reads <project>/tools_settings.json when it is imported unless
    SSN_TOOL_SETTINGS_SCRIPT/FILE name another file; main() then applies the
    file given on the command line.
    """
    return {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "SSN_TOOL_SETTINGS_SCRIPT": "Sanitize_Sequences.py",
        "SSN_TOOL_SETTINGS_FILE": os.path.join(temp_dir, "no-shared-settings.json"),
    }


class SanitizeSequencesTests(unittest.TestCase):
    def test_length_table_matches_histogram_counts(self):
        for lengths in ([1, 2, 2, 26, 51], [10, 10], [7]):
            with self.subTest(lengths=lengths):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    sanitize_sequences.print_length_distribution(lengths)
                rows = [line for line in output.getvalue().splitlines() if line.startswith("[")]
                counts, edges = sanitize_sequences.np.histogram(lengths, bins=50)
                self.assertEqual(len(rows), 50)
                self.assertEqual(
                    [int(row.split("|")[1]) for row in rows], counts.tolist()
                )
                self.assertTrue(rows[0].startswith(f"[{edges[0]:.12g}, "))
                self.assertIn(f", {edges[-1]:.12g}]", rows[-1])

    def test_mcp_headless_invocation_never_imports_matplotlib(self):
        from tools.tool_helpers.Tool_Pipeline import prepare_headless_invocation

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = pathlib.Path(temp_dir) / "input.fasta"
            input_path.write_text(">short\nAC\n>long\nACDEF\n", encoding="utf-8")
            invocation = prepare_headless_invocation(
                "sanitize_sequences",
                {
                    "DIRECTORIES": {"FASTA_DIR": temp_dir},
                    "Sanitize_Sequences.py": {
                        "INPUT_FASTA": str(input_path),
                        "OVER_WRITE": False,
                        "ENABLE_LENGTH_FILTER": False,
                        "REMOVE_BY_HEADER_STRING": "",
                    },
                },
                PROJECT_ROOT,
                snapshot_directory=temp_dir,
            )
            # Block even importing the plotting package in a fresh interpreter.
            code = (
                "import runpy, sys; "
                "sys.modules['matplotlib'] = None; "
                f"sys.path.insert(0, {str(MODULE_PATH.parent)!r}); "
                f"sys.argv = {list(invocation.argv[2:])!r}; "
                f"runpy.run_path({str(MODULE_PATH)!r}, run_name='__main__')"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=invocation.cwd,
                env=_tool_environment(temp_dir),
                capture_output=True, text=True, encoding="utf-8", timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Frequency (sequences)", result.stdout)
            self.assertNotIn("Opening length distribution histogram", result.stdout)
            self.assertEqual(
                (pathlib.Path(temp_dir) / "input_sanitized.fasta").read_text(),
                ">short\nAC\n>long\nACDEF\n",
            )

    def test_utf8_inputs_match_downstream_and_write_without_bom(self):
        from utilities.Sequence_Utils import load_sanitized_fasta

        for encoding in ("utf-8", "utf-8-sig"):
            for overwrite in (False, True):
                with self.subTest(encoding=encoding, overwrite=overwrite):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        input_path = pathlib.Path(temp_dir) / "input.fasta"
                        input_path.write_text(
                            ">long descriptive header\nACDE\n"
                            ">short\nACDE\n>unique\nFGHI\n",
                            encoding=encoding,
                        )
                        expected_headers, expected_sequences, _ = load_sanitized_fasta(
                            input_path, report=False,
                        )
                        settings_path = pathlib.Path(temp_dir) / "settings.json"
                        settings_path.write_text(json.dumps({
                            "DIRECTORIES": {"FASTA_DIR": temp_dir},
                            "Sanitize_Sequences.py": {
                                "INPUT_FASTA": str(input_path),
                                "OVER_WRITE": overwrite,
                                "ENABLE_LENGTH_FILTER": False,
                                "REMOVE_BY_HEADER_STRING": "",
                            },
                        }), encoding="utf-8")

                        result = subprocess.run(
                            [sys.executable, "-B", str(MODULE_PATH), str(settings_path)],
                            cwd=PROJECT_ROOT,
                            env=_tool_environment(temp_dir),
                            capture_output=True, text=True, encoding="utf-8", timeout=30,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        output_path = (
                            input_path if overwrite
                            else pathlib.Path(temp_dir) / "input_sanitized.fasta"
                        )
                        output_bytes = output_path.read_bytes()
                        self.assertFalse(output_bytes.startswith(b"\xef\xbb\xbf"))
                        self.assertEqual(
                            output_bytes.decode("utf-8"),
                            ">long_descriptive_header\nACDE\n>unique\nFGHI\n",
                        )
                        self.assertEqual(
                            sanitize_sequences.read_fasta(output_path),
                            (expected_headers, expected_sequences),
                        )

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

    def test_configuration_requires_an_explicit_output_directory(self):
        with self.assertRaisesRegex(ValueError, "output directory"):
            sanitize_sequences.validate_configuration(
                None,
                "input.fasta",
                "output.fasta",
            )

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


if __name__ == "__main__":
    unittest.main()
