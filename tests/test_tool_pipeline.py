"""Pipeline tool catalog and invocations (tools.tool_helpers.Tool_Pipeline).

Covers the stable tool catalog, the GUI command line EMAPSSN_Tools runs, and
the settings snapshots a headless invocation writes, plus the quiet import of
the transport-neutral modules the MCP server and the Viewer share.
"""
import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tools.tool_helpers.Tool_Pipeline import (  # noqa: E402
    format_invocation_command,
    get_tool_spec,
    list_tool_specs,
    prepare_gui_invocation,
    prepare_headless_invocation,
)


class ToolExecutionTests(unittest.TestCase):
    def test_transport_neutral_modules_import_without_stdout(self):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(SRC_DIR)!r}); "
            "import tools.tool_helpers.Tool_Pipeline; "
            "import desktop.Viewer_Inspection; "
            "import utilities.Viewer_Sessions"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_catalog_is_complete_stable_and_quiet(self):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            specs = list_tool_specs()
        self.assertEqual(captured.getvalue(), "")
        self.assertEqual(len(specs), 14)
        self.assertEqual(len({spec.tool_id for spec in specs}), 14)
        self.assertTrue(all(spec.output_directories for spec in specs))
        self.assertTrue(
            all(
                set(spec.output_directories).issubset(spec.required_directories)
                for spec in specs
            )
        )
        self.assertEqual(
            get_tool_spec("sanitize_sequences").script_name,
            "Sanitize_Sequences.py",
        )
        with self.assertRaises(KeyError):
            get_tool_spec("../arbitrary.py")

    def test_gui_invocation_preserves_the_existing_command(self):
        script = SRC_DIR / "tools" / "Sanitize_Sequences.py"
        invocation = prepare_gui_invocation(
            script,
            PROJECT_ROOT,
            python_executable="managed-python",
        )
        self.assertEqual(
            invocation.argv,
            ("managed-python", "-u", str(script.resolve())),
        )
        self.assertEqual(invocation.cwd, str(script.parent.resolve()))
        self.assertFalse(invocation.owns_settings_snapshot)
        self.assertEqual(
            format_invocation_command(invocation),
            f'"managed-python" -u "{script.resolve()}"',
        )

    def test_object_and_file_settings_create_equivalent_snapshots(self):
        document = {
            "DIRECTORIES": {"FASTA_DIR": "Input_Files/Sequence_Sets"},
            "Sanitize_Sequences.py": {
                "INPUT_FASTA": None,
                "OVER_WRITE": False,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = pathlib.Path(temp_dir) / "source.json"
            source_path.write_text(json.dumps(document), encoding="utf-8")
            object_call = prepare_headless_invocation(
                "sanitize_sequences",
                document,
                PROJECT_ROOT,
                python_executable="managed-python",
                snapshot_directory=temp_dir,
            )
            file_call = prepare_headless_invocation(
                "sanitize_sequences",
                source_path,
                PROJECT_ROOT,
                python_executable="managed-python",
                snapshot_directory=temp_dir,
            )
            object_payload = json.loads(
                pathlib.Path(object_call.settings_path).read_text(encoding="utf-8")
            )
            file_payload = json.loads(
                pathlib.Path(file_call.settings_path).read_text(encoding="utf-8")
            )
            self.assertEqual(object_payload, document)
            self.assertEqual(file_payload, document)
            self.assertIsNone(
                object_payload["Sanitize_Sequences.py"]["INPUT_FASTA"]
            )
            self.assertEqual(object_call.argv[-1], object_call.settings_path)
            self.assertTrue(object_call.owns_settings_snapshot)


if __name__ == "__main__":
    unittest.main()
