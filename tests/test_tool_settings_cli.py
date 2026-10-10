"""Tool settings: loading, saving and exporting the settings documents tools run from.

Covers tools.tool_helpers.Tool_Pipeline (settings-file selection, validation,
application to a tool's globals, the hand-off to spawned workers, the shared
tools_settings.json and its directory defaults), the tool entry points, and the
portable directory form of exported settings (utilities.Headless_Settings).
"""
import ast
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import ntpath
import os
import pathlib
import posixpath
import subprocess
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools.tool_helpers.Tool_Pipeline import (  # noqa: E402
    DEFAULT_DIRECTORY_PATHS,
    TOOL_DIRECTORY_KEYS,
    ToolSettingsError,
    apply_settings_document,
    fill_missing_directory_defaults,
    inherited_settings_path,
    load_shared_settings,
    load_tool_settings,
    project_directory_defaults,
    read_settings_document,
    read_shared_settings,
    save_shared_directories,
    save_shared_tool_settings,
    select_settings_path,
)


EXPECTED_TOOLS = {
    "Align_Similarity_Matrix.py",
    "Align_Substitution_Matrix.py",
    "Embedding_Cropping.py",
    "Embedding_Extraction.py",
    "Embedding_Injection.py",
    "Embedding_MSA.py",
    "Embedding_PWA.py",
    "Embedding_SSEARCH.py",
    "Generate_Embeddings.py",
    "Network_Extraction.py",
    "Network_Injection.py",
    "Parse_BLAST_Output.py",
    "Sanitize_Sequences.py",
    "Sparse_MSA_Converter.py",
}


class ToolSettingsLoaderTests(unittest.TestCase):
    def test_registry_covers_every_gui_tool_and_export_default(self):
        self.assertEqual(set(TOOL_DIRECTORY_KEYS), EXPECTED_TOOLS)
        self.assertEqual(
            TOOL_DIRECTORY_KEYS["Network_Extraction.py"],
            ("FASTA_DIR", "NETWORK_DIR"),
        )
        self.assertEqual(
            TOOL_DIRECTORY_KEYS["Parse_BLAST_Output.py"],
            ("FASTA_DIR", "NETWORK_DIR"),
        )
        self.assertNotIn("PATH_DIR", DEFAULT_DIRECTORY_PATHS)
        self.assertEqual(
            DEFAULT_DIRECTORY_PATHS["SETTING_EXPORT_DIR"],
            os.path.join("Cache_Files", "Exported_Settings"),
        )

    def test_explicit_document_applies_types_and_project_relative_paths(self):
        namespace = {
            "FASTA_DIR": "default",
            "INPUT_FASTA": None,
            "COUNT": 1,
            "RATIO": 1.0,
            "ENABLED": False,
            "OPTIONAL": None,
            "SAFE_TEMP_DIR": "default",
        }
        document = {
            "DIRECTORIES": {"FASTA_DIR": os.path.join("Input_Files", "Sequences")},
            "Example.py": {
                "INPUT_FASTA": "input.fasta",
                "COUNT": "7",
                "RATIO": "2.5",
                "ENABLED": True,
                "OPTIONAL": "None",
                "SAFE_TEMP_DIR": os.path.join("Cache_Files", "Temp"),
            },
        }

        apply_settings_document(namespace, document, "Example.py", str(PROJECT_ROOT))

        self.assertEqual(
            namespace["FASTA_DIR"],
            os.path.normpath(PROJECT_ROOT / "Input_Files" / "Sequences"),
        )
        self.assertEqual(namespace["INPUT_FASTA"], "input.fasta")
        self.assertEqual(namespace["COUNT"], 7)
        self.assertEqual(namespace["RATIO"], 2.5)
        self.assertIs(namespace["ENABLED"], True)
        self.assertIsNone(namespace["OPTIONAL"])
        self.assertEqual(
            namespace["SAFE_TEMP_DIR"],
            os.path.normpath(PROJECT_ROOT / "Cache_Files" / "Temp"),
        )

    def test_explicit_path_is_resolved_from_working_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            current = pathlib.Path.cwd()
            try:
                os.chdir(temp_dir)
                path, explicit = select_settings_path(
                    "Example.py", str(PROJECT_ROOT), ["profile.json"]
                )
            finally:
                os.chdir(current)
            self.assertTrue(explicit)
            self.assertEqual(path, os.path.join(temp_dir, "profile.json"))

    def test_no_argument_falls_back_to_shared_settings(self):
        path, explicit = select_settings_path("Example.py", str(PROJECT_ROOT), [])
        self.assertFalse(explicit)
        self.assertEqual(
            path,
            os.path.join(PROJECT_ROOT, "tools_settings.json"),
        )

    def test_spawn_inheritance_is_scoped_to_the_originating_tool(self):
        with mock.patch.dict(
            os.environ,
            {
                "SSN_TOOL_SETTINGS_FILE": "C:/portable/settings.json",
                "SSN_TOOL_SETTINGS_SCRIPT": "Example.py",
            },
        ):
            self.assertEqual(
                inherited_settings_path("C:/project/tools/Example.py"),
                "C:/portable/settings.json",
            )
            self.assertIsNone(
                inherited_settings_path("C:/project/tools/Other.py")
            )

    def test_missing_malformed_and_wrong_tool_explicit_files_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            missing = temp_path / "missing.json"
            malformed = temp_path / "malformed.json"
            malformed.write_text("{", encoding="utf-8")
            wrong = temp_path / "wrong.json"
            wrong.write_text(
                json.dumps({"DIRECTORIES": {}, "Other.py": {}}),
                encoding="utf-8",
            )

            for path in (missing, malformed, wrong):
                with self.subTest(path=path.name), self.assertRaises(SystemExit) as error:
                    load_tool_settings({}, "Example.py", str(PROJECT_ROOT), [str(path)])
                self.assertEqual(error.exception.code, 2)

    def test_extra_arguments_fail_before_loading(self):
        with self.assertRaises(SystemExit) as error:
            load_tool_settings(
                {}, "Example.py", str(PROJECT_ROOT), ["one.json", "two.json"]
            )
        self.assertEqual(error.exception.code, 2)

    def test_missing_shared_file_is_a_backward_compatible_empty_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document = read_settings_document(
                pathlib.Path(temp_dir) / "missing.json",
                "Example.py",
                explicit=False,
            )
        self.assertEqual(document, {})

    def test_an_unreadable_shared_file_stops_the_tool_and_is_kept(self):
        # Such a file used to print a warning and leave the tool on its defaults.
        unreadable = {
            "corrupt": ('{"Example.py": {"COUNT": "3"},}', r"line 1 column \d+"),
            "empty": ("", r"line 1 column 1 "),
            "not an object": ('[{"Example.py": {"COUNT": "3"}}]', "must contain a JSON object"),
        }
        for label, (content, reason) in unreadable.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as root, \
                    mock.patch.dict(os.environ):
                os.environ.pop("SSN_TOOL_SETTINGS_FILE", None)
                path = pathlib.Path(root) / "tools_settings.json"
                path.write_text(content, encoding="utf-8")
                namespace = {"COUNT": 1}
                errors = io.StringIO()
                with redirect_stderr(errors), self.assertRaises(SystemExit) as stopped:
                    load_tool_settings(namespace, "/x/tools/Example.py", root, [])

                self.assertEqual(stopped.exception.code, 2)
                self.assertIn(str(path), errors.getvalue())
                self.assertRegex(errors.getvalue(), reason)
                self.assertEqual(namespace, {"COUNT": 1})
                self.assertNotIn("SSN_TOOL_SETTINGS_FILE", os.environ)
                self.assertEqual(path.read_text(encoding="utf-8"), content)


class ToolEntryPointTests(unittest.TestCase):
    def test_a_script_run_stops_on_an_unreadable_shared_file_before_its_inputs(self):
        # Tools only ever run as scripts: exec one as main() would start, in a
        # stand-in project whose tools_settings.json is written afterwards.
        tool = SRC_DIR / "tools" / "Sanitize_Sequences.py"
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ):
            for name in ("SSN_TOOL_SETTINGS_SCRIPT", "SSN_TOOL_SETTINGS_FILE"):
                os.environ.pop(name, None)
            project = pathlib.Path(temp) / "project"
            namespace = {"__name__": "sanitize_as_script",
                         "__file__": str(project / "src" / "tools" / tool.name)}
            exec(compile(tool.read_text(encoding="utf-8"), str(tool), "exec"), namespace)
            project.mkdir()
            settings = project / "tools_settings.json"
            content = '{"Sanitize_Sequences.py": {"INPUT_FASTA": "input.fasta"},}'
            settings.write_text(content, encoding="utf-8")

            output, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(errors), \
                    self.assertRaises(SystemExit) as stopped:
                namespace["main"]([])

            self.assertEqual(stopped.exception.code, 2)
            self.assertIn(str(settings), errors.getvalue())
            self.assertRegex(errors.getvalue(), r"line 1 column \d+")
            self.assertNotIn("Reading from", output.getvalue())
            self.assertEqual(os.listdir(project), ["tools_settings.json"])
            self.assertEqual(settings.read_text(encoding="utf-8"), content)

    def test_every_registered_tool_exposes_main_and_uses_shared_loader(self):
        for filename in EXPECTED_TOOLS:
            with self.subTest(tool=filename):
                source = (SRC_DIR / "tools" / filename).read_text(encoding="utf-8")
                self.assertIn("def main(argv=None):", source)
                self.assertIn("load_tool_settings(globals(), __file__, PROJECT_ROOT", source)

    def test_parse_blast_gui_contract_has_required_order_and_custom_gating(self):
        source = (SRC_DIR / "EMAPSSN_Tools.py").read_text(encoding="utf-8")
        manual_settings = source.index("self.MANUAL_SETTINGS")
        start = source.index('"Parse_BLAST_Output.py": [', manual_settings)
        end = source.index('"Embedding_MSA": {', start)
        panel = source[start:end]
        expected_order = (
            '"var_name": "INPUT_BLAST_TABULAR"',
            '"var_name": "INPUT_FASTA"',
            '"var_name": "BLAST_LAYOUT"',
            '"var_name": "QUERY_COLUMN"',
            '"var_name": "SUBJECT_COLUMN"',
            '"var_name": "EVALUE_COLUMN"',
        )
        positions = [panel.index(token) for token in expected_order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('(".tabular", ".txt", ".tab", ".tsv")', panel)
        self.assertIn('translate("Tools", "Custom Columns (1-based indexing)")', panel)
        self.assertIn('"display": translate("Tools", "Query Column:")', panel)
        self.assertIn('"display": translate("Tools", "Subject Column:")', panel)
        self.assertIn('"display": translate("Tools", "EValue Column:")', panel)
        self.assertNotIn('"var_name": "MATRIX"', panel)
        self.assertNotIn('"var_name": "BATCH_SIZE"', panel)
        self.assertIn("bind_custom_blast_column_controls(inputs, row_widgets)", source)

    def test_representative_main_receives_explicit_export_before_worker(self):
        module_path = SRC_DIR / "tools" / "Align_Similarity_Matrix.py"
        spec = importlib.util.spec_from_file_location("cli_alignment_tool", module_path)
        module = importlib.util.module_from_spec(spec)
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ,
            {
                "SSN_TOOL_SETTINGS_SCRIPT": "Align_Similarity_Matrix.py",
                "SSN_TOOL_SETTINGS_FILE": os.path.join(temp_dir, "missing.json"),
            },
        ):
            spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = pathlib.Path(temp_dir) / "alignment.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "DIRECTORIES": {
                            "EMBED_DIR": "portable_embeddings",
                            "NETWORK_DIR": "portable_networks",
                        },
                        "Align_Similarity_Matrix.py": {
                            "INPUT_HDF5": "portable.h5",
                            "WORKERS": 3,
                            "EXECUTION_MODE": "tiled",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(
                module, "run_job_distributor", return_value=None
            ) as worker:
                result = module.main([str(settings_path)])

        self.assertEqual(result, 0)
        worker.assert_called_once_with()
        self.assertEqual(module.INPUT_HDF5, "portable.h5")
        self.assertEqual(module.WORKERS, 3)
        self.assertEqual(module.EXECUTION_MODE, "tiled")
        self.assertEqual(
            module.EMBED_DIR,
            os.path.normpath(PROJECT_ROOT / "portable_embeddings"),
        )


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

    def test_shared_execution_populates_missing_global_directories(self):
        gui_source = (PROJECT_ROOT / "src" / "EMAPSSN_Tools.py").read_text(
            encoding="utf-8"
        )
        gui_tree = ast.parse(gui_source)
        save_and_run = next(
            node
            for node in ast.walk(gui_tree)
            if isinstance(node, ast.FunctionDef) and node.name == "save_and_run"
        )
        gui_calls = {
            node.func.id
            for node in ast.walk(save_and_run)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("load_shared_settings", gui_calls)
        self.assertIn("save_shared_tool_settings", gui_calls)

        service_source = (
            PROJECT_ROOT / "src" / "tools" / "tool_helpers" / "Tool_Pipeline.py"
        ).read_text(encoding="utf-8")
        service_tree = ast.parse(service_source)
        shared_functions = {
            node.name: node
            for node in ast.walk(service_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in {"load_shared_settings", "save_shared_tool_settings"}
        }
        self.assertEqual(
            set(shared_functions),
            {"load_shared_settings", "save_shared_tool_settings"},
        )
        for function in shared_functions.values():
            calls = {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            self.assertIn("fill_missing_directory_defaults", calls)

    def test_every_tool_uses_the_shared_project_anchored_fallbacks(self):
        tools_dir = PROJECT_ROOT / "src" / "tools"
        for filename, expected_directories in TOOL_DIRECTORY_KEYS.items():
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


class ToolSettingsHandOffTests(unittest.TestCase):
    """load_tool_settings names the file it used for the workers its tool spawns."""

    def test_explicit_file_is_handed_to_workers_of_the_same_tool_only(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(os.environ):
            pathlib.Path(temp_dir, "export.json").write_text(
                json.dumps({"DIRECTORIES": {}, "Example.py": {"COUNT": "3"}}),
                encoding="utf-8",
            )
            namespace = {"COUNT": 1}
            current = pathlib.Path.cwd()
            try:
                os.chdir(temp_dir)
                returned = load_tool_settings(
                    namespace, "/x/tools/Example.py", str(PROJECT_ROOT), ["export.json"]
                )
            finally:
                os.chdir(current)

            expected = os.path.join(temp_dir, "export.json")
            self.assertEqual(returned, expected)
            self.assertEqual(namespace["COUNT"], 3)
            self.assertEqual(os.environ["SSN_TOOL_SETTINGS_FILE"], expected)
            self.assertEqual(os.environ["SSN_TOOL_SETTINGS_SCRIPT"], "Example.py")
            self.assertEqual(inherited_settings_path("/x/Example.py"), expected)
            self.assertIsNone(inherited_settings_path("/x/Other.py"))

    def test_without_an_argument_the_shared_file_is_handed_on(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ):
            returned = load_tool_settings({}, "/x/tools/Example.py", root, [])

            expected = os.path.join(root, "tools_settings.json")
            self.assertEqual(returned, expected)
            self.assertEqual(os.environ["SSN_TOOL_SETTINGS_FILE"], expected)
            self.assertEqual(os.environ["SSN_TOOL_SETTINGS_SCRIPT"], "Example.py")
            self.assertEqual(inherited_settings_path("/elsewhere/Example.py"), expected)


def _module_level_statements(tree):
    """Yield the statements that run when a module is imported."""
    pending = list(tree.body)
    while pending:
        node = pending.pop(0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield node
        for field in ("body", "orelse", "finalbody", "handlers"):
            pending.extend(getattr(node, field, []))


class ToolWorkerImportTests(unittest.TestCase):
    """A spawned worker imports its tool and must see the settings main() used."""

    def test_no_tool_imports_another_tool_when_it_is_imported(self):
        # Another tool's import-time block would read the shared
        # tools_settings.json, since the hand-off names only the running tool.
        tool_names = {name[:-3] for name in EXPECTED_TOOLS}
        found = []
        for path in sorted((SRC_DIR / "tools").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in _module_level_statements(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    if module.removeprefix("tools.") in tool_names:
                        found.append(f"{path.name}:{node.lineno} imports {module}")
        self.assertEqual(found, [])


class SharedToolSettingsTests(unittest.TestCase):
    """The Tools window's writers rewrite one section of tools_settings.json.

    save_shared_tool_settings replaces a tool's section and
    save_shared_directories replaces DIRECTORIES; neither ever writes over a
    file it cannot load, and both replace the file atomically.
    """

    @staticmethod
    def write_shared(root, text):
        path = pathlib.Path(root) / "tools_settings.json"
        path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def saves(root):
        """Each shared writer, applied to the tools_settings.json in root."""
        return {
            "tool section": lambda: save_shared_tool_settings(
                root, "sanitize_sequences", {"INPUT_FASTA": "new.fasta"}
            ),
            "directories": lambda: save_shared_directories(root, {"FASTA_DIR": "new"}),
        }

    def test_saving_replaces_one_section_keeps_the_others_and_fills_directories(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_shared(root, json.dumps({
                "DIRECTORIES": {"FASTA_DIR": "custom", "MSA_DIR": "  "},
                "Sanitize_Sequences.py": {"STALE": 1, "INPUT_FASTA": "old.fasta"},
                "Generate_Embeddings.py": {"MODEL_NAME": "esm2_t6_8m", "BATCH_SIZE": 4},
            }))

            returned = save_shared_tool_settings(
                root, "sanitize_sequences", {"INPUT_FASTA": "new.fasta", "OVER_WRITE": True}
            )
            document = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(returned, str(path))
        self.assertEqual(document, {
            "DIRECTORIES": {**DEFAULT_DIRECTORY_PATHS, "FASTA_DIR": "custom"},
            "Sanitize_Sequences.py": {"INPUT_FASTA": "new.fasta", "OVER_WRITE": True},
            "Generate_Embeddings.py": {"MODEL_NAME": "esm2_t6_8m", "BATCH_SIZE": 4},
        })

    def test_reading_keeps_saved_values_and_loading_fills_directories(self):
        saved = {"DIRECTORIES": {"FASTA_DIR": "", "MSA_DIR": "custom"}, "Embedding_MSA.py": {}}
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(read_shared_settings(root), {})
            self.write_shared(root, json.dumps(saved))

            self.assertEqual(read_shared_settings(root), saved)
            self.assertEqual(
                load_shared_settings(root)["DIRECTORIES"],
                {**DEFAULT_DIRECTORY_PATHS, "MSA_DIR": "custom"},
            )

    def test_saving_directories_replaces_only_that_section(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_shared(root, json.dumps({
                "DIRECTORIES": {"FASTA_DIR": "old", "PATH_DIR": "legacy"},
                "Generate_Embeddings.py": {"MODEL_NAME": "esm2_t6_8m"},
                "Embedding_MSA.py": {"TREE_METHOD": "UPGMA"},
            }))

            returned = save_shared_directories(root, {"FASTA_DIR": "new", "MSA_DIR": ""})
            document = json.loads(path.read_text(encoding="utf-8"))

            path.unlink()
            save_shared_directories(root, {"FASTA_DIR": "new"})
            created = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(returned, str(path))
        self.assertEqual(document, {
            "DIRECTORIES": {"FASTA_DIR": "new", "MSA_DIR": ""},
            "Generate_Embeddings.py": {"MODEL_NAME": "esm2_t6_8m"},
            "Embedding_MSA.py": {"TREE_METHOD": "UPGMA"},
        })
        self.assertEqual(created, {"DIRECTORIES": {"FASTA_DIR": "new"}})

    def test_a_file_that_cannot_be_loaded_is_reported_and_kept(self):
        # One trailing comma used to make the document count as empty, so a
        # save kept only the saved section and wiped every other tool's.
        unreadable = {
            "trailing comma": (
                b'{"Generate_Embeddings.py": {"MODEL_NAME": "esm2_t6_8m"},}',
                r"line 1 column \d+",
            ),
            "array root": (b"[1, 2]", "must contain a JSON object"),
            "not UTF-8": (b'{"DIRECTORIES": "\xff"}', "codec can't decode"),
        }
        for label, (content, reason) in unreadable.items():
            with tempfile.TemporaryDirectory() as root:
                path = pathlib.Path(root) / "tools_settings.json"
                path.write_bytes(content)
                for read in (read_shared_settings, load_shared_settings):
                    with self.subTest(label, read=read.__name__), self.assertRaisesRegex(
                        ToolSettingsError, reason
                    ):
                        read(root)
                for writer, save in self.saves(root).items():
                    with self.subTest(label, writer=writer):
                        with self.assertRaisesRegex(ToolSettingsError, "tools_settings.json"):
                            save()
                        self.assertEqual(path.read_bytes(), content)

    def test_a_failed_write_leaves_the_previous_file_in_place(self):
        original = json.dumps({"Generate_Embeddings.py": {"MODEL_NAME": "esm2_t6_8m"}})

        def fail_midway(document, handle, **options):
            handle.write('{"DIRECTORIES": ')
            raise OSError(28, "No space left on device")

        for writer in ("tool section", "directories"):
            with self.subTest(writer=writer), tempfile.TemporaryDirectory() as root:
                path = self.write_shared(root, original)
                with mock.patch.object(json, "dump", side_effect=fail_midway), \
                        self.assertRaisesRegex(OSError, "No space left"):
                    self.saves(root)[writer]()
                self.assertEqual(path.read_text(encoding="utf-8"), original)
                self.assertEqual(os.listdir(root), ["tools_settings.json"])

    def test_an_interrupted_save_copy_cannot_be_committed(self):
        class Killed(BaseException):
            pass

        # A hard kill between writing the copy and the rename leaves the copy.
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(os, "replace", side_effect=Killed), \
                    mock.patch.object(os, "unlink"), self.assertRaises(Killed):
                self.saves(root)["tool section"]()
            [leftover] = os.listdir(root)
        try:
            result = subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), "check-ignore", "--no-index", "--quiet", "--", leftover],
                capture_output=True,
            )
        except OSError:
            self.skipTest("git is not installed")
        if result.returncode not in (0, 1):
            self.skipTest("not a git checkout")
        self.assertEqual(result.returncode, 0, f"{leftover} is not git-ignored")


class PortableExportDirectoryTests(unittest.TestCase):
    """Exported settings keep the GUI's portable form of each directory."""

    def test_exported_relative_directories_use_portable_separators(self):
        from utilities.Headless_Settings import build_pipeline_export

        def portable(directory):
            document = build_pipeline_export(
                "sanitize_sequences",
                {"FASTA_DIR": directory},
                {},
                str(PROJECT_ROOT),
                absolute=False,
            )
            return document["DIRECTORIES"]["FASTA_DIR"]

        exported = portable(r"Input_Files\Sequence Sets")
        self.assertEqual(exported, "Input_Files/Sequence Sets")
        self.assertEqual(
            portable("Input_Files/Sequence Sets"),
            "Input_Files/Sequence Sets",
        )
        self.assertEqual(
            ntpath.normpath(ntpath.join(r"C:\ssn", exported)),
            r"C:\ssn\Input_Files\Sequence Sets",
        )
        self.assertEqual(
            posixpath.normpath(posixpath.join("/ssn", exported)),
            "/ssn/Input_Files/Sequence Sets",
        )
        self.assertEqual(portable(r"C:\SSN Data\Sequences"), r"C:\SSN Data\Sequences")
        self.assertEqual(portable("/srv/ssn/sequences"), "/srv/ssn/sequences")


if __name__ == "__main__":
    unittest.main()
