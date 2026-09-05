import importlib.util
import json
import ntpath
import os
import pathlib
import posixpath
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from utilities.Tool_Directories import (  # noqa: E402
    DEFAULT_DIRECTORY_PATHS,
    TOOL_DIRECTORY_KEYS,
)
from utilities.Tool_Settings import (  # noqa: E402
    apply_settings_document,
    inherited_settings_path,
    load_tool_settings,
    read_settings_document,
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
            with mock.patch("os.getcwd", return_value=temp_dir):
                old_cwd = os.getcwd()
            current = pathlib.Path.cwd()
            try:
                os.chdir(temp_dir)
                path, explicit = select_settings_path(
                    "Example.py", str(PROJECT_ROOT), ["profile.json"]
                )
            finally:
                os.chdir(current)
            self.assertTrue(explicit)
            self.assertEqual(path, os.path.join(old_cwd, "profile.json"))

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


class ToolEntryPointTests(unittest.TestCase):
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
        self.assertIn('"Custom Columns (1-based indexing)"', panel)
        self.assertIn('"display": "Query Column:"', panel)
        self.assertIn('"display": "Subject Column:"', panel)
        self.assertIn('"display": "EValue Column:"', panel)
        self.assertNotIn('"var_name": "MATRIX"', panel)
        self.assertNotIn('"var_name": "BATCH_SIZE"', panel)
        self.assertIn(
            '("QUERY_COLUMN", "SUBJECT_COLUMN", "EVALUE_COLUMN")', source
        )
        self.assertIn("bind_custom_blast_column_controls(inputs, row_widgets)", source)

    def test_parse_blast_uses_fixed_import_metadata_and_batch_size(self):
        module_path = SRC_DIR / "tools" / "Parse_BLAST_Output.py"
        spec = importlib.util.spec_from_file_location("fixed_blast_parser", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = pathlib.Path(temp_dir) / "parser.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "DIRECTORIES": {
                            "FASTA_DIR": temp_dir,
                            "NETWORK_DIR": temp_dir,
                        },
                        "Parse_BLAST_Output.py": {
                            "INPUT_BLAST_TABULAR": "input.tabular",
                            "INPUT_FASTA": "input.fasta",
                            "MATRIX": "BLOSUM62",
                            "BATCH_SIZE": 7,
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary = SimpleNamespace(
                fasta_header_count=2,
                data_rows=1,
                self_rows=0,
                unique_edges=1,
                output_path=str(pathlib.Path(temp_dir) / "output.h5"),
            )
            with mock.patch.object(
                module, "build_blast_network", return_value=summary
            ) as builder:
                self.assertEqual(module.main([str(settings_path)]), 0)

        self.assertEqual(builder.call_args.kwargs["matrix"], "Imported")
        self.assertEqual(builder.call_args.kwargs["batch_size"], 1000000)

    def test_representative_main_receives_explicit_export_before_worker(self):
        module_path = SRC_DIR / "tools" / "Align_Similarity_Matrix.py"
        spec = importlib.util.spec_from_file_location("cli_alignment_tool", module_path)
        module = importlib.util.module_from_spec(spec)
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
                module, "run_job_distributor"
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


class ToolExportGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        from EMAPSSN_Tools import (
            HostCacheControl,
            ToolsGUI,
            _selection_supports_bf16,
            _selection_supports_tf32,
            _sync_alignment_tiled_option,
            _sync_tf32_precision_option,
        )

        cls.app = QApplication.instance() or QApplication([])
        cls.host_cache_control_class = HostCacheControl
        cls.tools_gui_class = ToolsGUI
        cls.selection_supports_tf32 = staticmethod(_selection_supports_tf32)
        cls.selection_supports_bf16 = staticmethod(_selection_supports_bf16)
        cls.sync_alignment_tiled_option = staticmethod(
            _sync_alignment_tiled_option
        )
        cls.sync_tf32_precision_option = staticmethod(
            _sync_tf32_precision_option
        )

    def test_export_filename_validation(self):
        normalize = self.tools_gui_class._normalized_export_filename
        self.assertEqual(normalize("analysis"), "analysis.json")
        self.assertEqual(normalize("analysis.JSON"), "analysis.JSON")
        for invalid in ("", "../escape", "bad:name", "CON", "CON.txt", "trailing."):
            with self.subTest(name=invalid), self.assertRaises(ValueError):
                normalize(invalid)

    def test_exported_relative_directories_use_portable_separators(self):
        portable = self.tools_gui_class._portable_export_directory_path

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

    def test_execution_mode_gui_contract_and_export_round_trip(self):
        from PySide6.QtWidgets import QComboBox, QInputDialog, QLineEdit, QMessageBox

        source = (SRC_DIR / "EMAPSSN_Tools.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count('"var_name": "EXECUTION_MODE"'), 2)
        self.assertGreaterEqual(
            source.count('"options": ["auto", "scalar", "tiled"]'), 2
        )

        script_path = str(SRC_DIR / "tools" / "Align_Similarity_Matrix.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            mode = QComboBox()
            mode.addItems(["auto", "scalar", "tiled"])
            mode.setCurrentText("tiled")
            fake_window = SimpleNamespace()
            fake_window.dir_inputs = {
                "EMBED_DIR": QLineEdit("Embeddings"),
                "NETWORK_DIR": QLineEdit(r"Input_Files\Networks_EValues"),
                "SETTING_EXPORT_DIR": QLineEdit(temp_dir),
            }
            fake_window.script_data = {
                script_path: {
                    "inputs": {
                        "EXECUTION_MODE": {
                            "widget": mode,
                            "type": "dropdown",
                        }
                    },
                    "settings": [{"name": "EXECUTION_MODE"}],
                }
            }
            fake_window._normalized_export_filename = (
                self.tools_gui_class._normalized_export_filename
            )
            fake_window._current_directory_settings = lambda: (
                self.tools_gui_class._current_directory_settings(fake_window)
            )
            fake_window._portable_export_directory_path = (
                self.tools_gui_class._portable_export_directory_path
            )
            fake_window._collect_tool_settings = lambda path: (
                self.tools_gui_class._collect_tool_settings(fake_window, path)
            )

            with mock.patch.object(
                QInputDialog, "getText", return_value=("alignment-mode", True)
            ), mock.patch.object(QMessageBox, "information"), mock.patch.object(
                QMessageBox, "critical"
            ) as critical:
                self.tools_gui_class.export_settings(fake_window, script_path)

            payload = json.loads(
                (pathlib.Path(temp_dir) / "alignment-mode.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                payload["Align_Similarity_Matrix.py"]["EXECUTION_MODE"],
                "tiled",
            )
            critical.assert_not_called()

    def test_tf32_precision_option_tracks_detected_and_selected_hardware(self):
        from PySide6.QtWidgets import QComboBox
        from utilities import Hardware_Utils
        import torch

        cpu = Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        cuda = Hardware_Utils.DeviceCandidate(
            "cuda:0", "CUDA", torch.device("cuda:0"), "cuda"
        )
        device = QComboBox()
        device.addItem("Auto", "auto")
        device.addItem("CPU", "cpu")
        device.addItem("CUDA", "cuda:0")
        precision = QComboBox()
        precision.addItem("auto", "auto")
        precision.addItem("float32", "float32")
        precision.addItem("TF32 (Nvidia GPU Only)", "tf32")
        precision.setCurrentIndex(precision.findData("tf32"))

        with mock.patch(
            "EMAPSSN_Tools.is_nvidia_cuda",
            side_effect=lambda selected: selected.type == "cuda",
        ):
            self.assertFalse(
                self.sync_tf32_precision_option(device, precision, [cpu])
            )
            self.assertEqual(precision.currentText(), "auto")
            self.assertEqual(precision.findData("tf32"), -1)
            self.assertFalse(precision.property("tf32Available"))

            self.assertTrue(
                self.sync_tf32_precision_option(
                    device,
                    precision,
                    [cpu, cuda],
                )
            )
            self.assertGreaterEqual(precision.findData("tf32"), 0)
            self.assertEqual(
                precision.itemText(precision.findData("tf32")),
                "TF32 (Nvidia GPU Only)",
            )

            precision.setCurrentIndex(precision.findData("tf32"))
            device.setCurrentIndex(device.findData("cpu"))
            self.assertFalse(
                self.sync_tf32_precision_option(
                    device,
                    precision,
                    [cpu, cuda],
                )
            )
            self.assertEqual(precision.currentText(), "auto")
            self.assertEqual(precision.findData("tf32"), -1)

            device.setCurrentIndex(device.findData("cuda:0"))
            self.assertTrue(
                self.sync_tf32_precision_option(
                    device,
                    precision,
                    [cpu, cuda],
                )
            )
            self.assertGreaterEqual(precision.findData("tf32"), 0)

    def test_bf16_precision_option_tracks_runtime_capability(self):
        from PySide6.QtWidgets import QComboBox
        from utilities import Hardware_Utils
        import torch

        cpu = Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        cuda = Hardware_Utils.DeviceCandidate(
            "cuda:0", "CUDA", torch.device("cuda:0"), "cuda"
        )
        device = QComboBox()
        device.addItem("Auto", "auto")
        device.addItem("CPU", "cpu")
        device.addItem("CUDA", "cuda:0")
        precision = QComboBox()
        precision.addItem("Automatic 32-bit", "automatic_32bit")
        precision.addItem("float32", "float32")
        precision.addItem("BF16 (Low Precision)", "bf16")

        with mock.patch(
            "EMAPSSN_Tools.bf16_accelerator_support",
            side_effect=lambda selected: (
                selected.type == "cuda",
                "mock capability",
            ),
        ), mock.patch(
            "EMAPSSN_Tools.is_nvidia_cuda",
            side_effect=lambda selected: selected.type == "cuda",
        ):
            self.sync_tf32_precision_option(device, precision, [cpu])
            self.assertEqual(precision.findData("bf16"), -1)
            self.assertFalse(precision.property("bf16Available"))

            self.sync_tf32_precision_option(device, precision, [cpu, cuda])
            self.assertGreaterEqual(precision.findData("bf16"), 0)
            self.assertTrue(precision.property("bf16Available"))

            device.setCurrentIndex(device.findData("cpu"))
            self.sync_tf32_precision_option(device, precision, [cpu, cuda])
            self.assertEqual(precision.findData("bf16"), -1)
            self.assertFalse(precision.property("bf16Available"))

    def test_alignment_tiled_option_hides_for_mps_and_restores_for_xpu(self):
        from PySide6.QtWidgets import QComboBox
        from utilities import Hardware_Utils
        import torch

        cpu = Hardware_Utils.DeviceCandidate(
            "cpu", "CPU", torch.device("cpu"), "cpu"
        )
        mps = Hardware_Utils.DeviceCandidate(
            "mps", "MPS", torch.device("mps"), "mps"
        )
        xpu = Hardware_Utils.DeviceCandidate(
            "xpu:0", "XPU", torch.device("xpu:0"), "xpu"
        )
        device = QComboBox()
        device.addItem("Auto", "auto")
        device.addItem("MPS", "mps")
        device.addItem("XPU", "xpu:0")
        execution = QComboBox()
        execution.addItems(["auto", "scalar", "tiled"])

        with mock.patch(
            "EMAPSSN_Tools.tiled_accelerator_support",
            return_value=(True, "mock support"),
        ):
            execution.setCurrentText("tiled")
            self.assertFalse(
                self.sync_alignment_tiled_option(
                    device, execution, [cpu, mps]
                )
            )
            self.assertEqual(execution.currentText(), "auto")
            self.assertEqual(execution.findText("tiled"), -1)
            self.assertFalse(execution.property("tiledAvailable"))

            self.assertTrue(
                self.sync_alignment_tiled_option(
                    device, execution, [cpu, mps, xpu]
                )
            )
            self.assertGreaterEqual(execution.findText("tiled"), 0)

            device.setCurrentIndex(device.findData("mps"))
            self.assertFalse(
                self.sync_alignment_tiled_option(
                    device, execution, [cpu, mps, xpu]
                )
            )
            self.assertEqual(execution.findText("tiled"), -1)

            device.setCurrentIndex(device.findData("xpu:0"))
            self.assertTrue(
                self.sync_alignment_tiled_option(
                    device,
                    execution,
                    [cpu, mps, xpu],
                    allow_mps=True,
                )
            )
            self.assertGreaterEqual(execution.findText("tiled"), 0)

            device.setCurrentIndex(device.findData("mps"))
            self.assertTrue(
                self.sync_alignment_tiled_option(
                    device,
                    execution,
                    [cpu, mps, xpu],
                    allow_mps=True,
                )
            )
            self.assertGreaterEqual(execution.findText("tiled"), 0)

    def test_host_cache_control_uses_auto_or_linear_manual_gib(self):
        source = (SRC_DIR / "EMAPSSN_Tools.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('"type": "host_cache"'), 2)

        control = self.host_cache_control_class("auto")
        script_path = str(SRC_DIR / "tools" / "Align_Similarity_Matrix.py")
        fake_window = SimpleNamespace(
            script_data={
                script_path: {
                    "inputs": {
                        "HOST_CACHE_GB": {
                            "widget": control,
                            "type": "host_cache",
                        }
                    },
                    "settings": [{"name": "HOST_CACHE_GB"}],
                }
            }
        )

        try:
            self.assertTrue(control.auto_button.isChecked())
            self.assertFalse(control.slider.isEnabled())
            self.assertFalse(control.spinbox.isEnabled())
            self.assertEqual(control.slider.styleSheet(), "")
            self.assertEqual(
                self.tools_gui_class._collect_tool_settings(
                    fake_window, script_path
                )["HOST_CACHE_GB"],
                "auto",
            )

            control.auto_button.click()
            self.app.processEvents()
            self.assertFalse(control.auto_button.isChecked())
            self.assertTrue(control.slider.isEnabled())
            self.assertTrue(control.spinbox.isEnabled())

            control.spinbox.setValue(64.0)
            self.assertEqual(control.slider.value(), 640)
            self.assertEqual(
                self.tools_gui_class._collect_tool_settings(
                    fake_window, script_path
                )["HOST_CACHE_GB"],
                64,
            )

            control.slider.setValue(0)
            self.assertEqual(control.spinbox.value(), 0.0)
            self.assertEqual(control.setting_value(), 0)
        finally:
            control.close()

    def test_host_cache_slider_is_linear_across_the_full_range(self):
        control_class = self.host_cache_control_class
        minimum = control_class.gb_for_slider_position(0)
        midpoint = control_class.gb_for_slider_position(640)
        maximum = control_class.gb_for_slider_position(1280)

        self.assertAlmostEqual(minimum, 0.0)
        self.assertAlmostEqual(midpoint, 64.0)
        self.assertAlmostEqual(maximum, 128.0)
        self.assertEqual(control_class.slider_position_for_gb(64.0), 640)

        manual_control = control_class(32)
        try:
            self.assertFalse(manual_control.auto_button.isChecked())
            self.assertTrue(manual_control.slider.isEnabled())
            self.assertEqual(manual_control.setting_value(), 32)
        finally:
            manual_control.close()

    def test_alignment_and_injection_hardware_rows_follow_requested_order(self):
        from PySide6.QtWidgets import QFormLayout, QLabel, QLineEdit, QWidget

        source = (SRC_DIR / "EMAPSSN_Tools.py").read_text(encoding="utf-8")
        manual_start = source.index("self.MANUAL_SETTINGS =")
        align_start = source.index('"Align_Similarity_Matrix.py": [', manual_start)
        align_end = source.index('"Align_Substitution_Matrix.py": [', align_start)
        align_source = source[align_start:align_end]
        self.assertLess(
            align_source.index('"var_name": "ACCELERATOR_PRECISION"'),
            align_source.index('"var_name": "EXECUTION_MODE"'),
        )
        self.assertLess(
            align_source.index('"var_name": "EXECUTION_MODE"'),
            align_source.index('"var_name": "HOST_CACHE_GB"'),
        )

        injection_start = source.index('"Network_Injection.py": [', manual_start)
        injection_end = source.index('"Network_Extraction.py": [', injection_start)
        injection_source = source[injection_start:injection_end]
        self.assertLess(
            injection_source.index('"var_name": "EXECUTION_MODE"'),
            injection_source.index('"var_name": "HOST_CACHE_GB"'),
        )

        cases = (
            (
                "Align_Similarity_Matrix.py",
                (
                    ("DEVICE_SELECTION", "Device:"),
                    ("ACCELERATOR_PRECISION", "Precision:"),
                    ("EXECUTION_MODE", "Execution Mode:"),
                    ("HOST_CACHE_GB", "Host Cache (GiB):"),
                ),
                "compactRow_ACCELERATOR_PRECISION_EXECUTION_MODE",
                ["Precision:", "Execution Mode:"],
            ),
            (
                "Network_Injection.py",
                (
                    ("DEVICE_SELECTION", "Device:"),
                    ("EXECUTION_MODE", "Execution Mode:"),
                    ("HOST_CACHE_GB", "Host Cache (GiB):"),
                ),
                "compactRow_EXECUTION_MODE_DEVICE_SELECTION",
                ["Execution Mode:", "Device:"],
            ),
        )
        for script_name, definitions, compact_name, compact_labels in cases:
            with self.subTest(script=script_name):
                form_parent = QWidget()
                layout = QFormLayout(form_parent)
                row_widgets = {}
                for var_name, label_text in definitions:
                    label = QLabel(label_text)
                    field = QLineEdit()
                    layout.addRow(label, field)
                    row_widgets[var_name] = (label, field)

                self.tools_gui_class._merge_compact_rows(
                    layout,
                    script_name,
                    row_widgets,
                )
                compact = form_parent.findChild(QWidget, compact_name)
                self.assertIsNotNone(compact)
                self.assertEqual(
                    [label.text() for label in compact.findChildren(QLabel)],
                    compact_labels,
                )
                compact_row = layout.getWidgetPosition(compact)[0]
                host_row = layout.getWidgetPosition(
                    row_widgets["HOST_CACHE_GB"][0]
                )[0]
                self.assertEqual(host_row, layout.rowCount() - 1)
                self.assertEqual(compact_row, host_row - 1)
                form_parent.close()

    def test_tab_pages_share_content_width_without_resizing_tab_labels(self):
        from PySide6.QtWidgets import QScrollArea, QTabWidget, QWidget

        tabs = QTabWidget()
        original_content_widths = (400, 750, 550)
        for title, content_width in zip(
            ("Short", "Longest Tool Category", "Medium Tab"),
            original_content_widths,
        ):
            content = QWidget()
            content.setMinimumWidth(content_width)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(content)
            tabs.addTab(scroll, title)

        original_tab_widths = {
            tabs.tabBar().tabSizeHint(index).width()
            for index in range(tabs.count())
        }
        fake_window = SimpleNamespace(
            tabs=tabs,
            COMMON_TAB_VIEWPORT_MINIMUM_WIDTH=600,
        )
        common_width = self.tools_gui_class._harmonize_tab_page_widths(
            fake_window
        )

        self.assertEqual(common_width, max(original_content_widths))
        self.assertEqual(
            tabs.property("commonContentMinimumWidth"),
            common_width,
        )
        self.assertEqual(tabs.property("commonViewportMinimumWidth"), 600)
        self.assertGreater(len(original_tab_widths), 1)
        self.assertEqual(
            {
                tabs.tabBar().tabSizeHint(index).width()
                for index in range(tabs.count())
            },
            original_tab_widths,
        )
        for index in range(tabs.count()):
            scroll_page = tabs.widget(index)
            content_page = scroll_page.widget()
            self.assertEqual(scroll_page.minimumWidth(), 600)
            self.assertEqual(
                scroll_page.property("commonViewportMinimumWidth"),
                600,
            )
            self.assertEqual(content_page.minimumWidth(), common_width)
            self.assertEqual(
                content_page.property("commonContentMinimumWidth"),
                common_width,
            )

    def test_tool_headers_align_with_shared_field_start(self):
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import (
            QBoxLayout,
            QFormLayout,
            QFrame,
            QLabel,
            QLineEdit,
            QPushButton,
        )

        fake_window = SimpleNamespace(
            tool_titles={},
            save_and_run=lambda script_path: None,
            export_settings=lambda script_path: None,
        )
        cards = []
        layouts = []
        fields = []
        headers = []
        for label_text in (
            "Short:",
            "Normalized Noise Scale (0 to 0.1):",
        ):
            card = QFrame()
            layout = QFormLayout(card)
            layout.setHorizontalSpacing(30)
            header = self.tools_gui_class._create_tool_header(
                fake_window,
                "Sanitize_Sequences.py",
                str(SRC_DIR / "tools" / "Sanitize_Sequences.py"),
            )
            field = QLineEdit()
            layout.addRow(header)
            layout.addRow(QLabel(label_text), field)
            cards.append(card)
            layouts.append(layout)
            fields.append(field)
            headers.append(header)

        shared_label_width = self.tools_gui_class._align_form_label_columns(layouts)
        title_start_x = self.tools_gui_class._align_tool_card_headers(
            layouts,
            shared_label_width,
        )

        try:
            for card in cards:
                card.resize(1000, 100)
                card.show()
            self.app.processEvents()

            field_positions = {
                field.mapTo(card, QPoint(0, 0)).x()
                for card, field in zip(cards, fields)
            }
            title_positions = {
                header.findChild(QLabel, "toolTitle")
                .mapTo(card, QPoint(0, 0))
                .x()
                for card, header in zip(cards, headers)
            }
            self.assertEqual(len(field_positions), 1)
            self.assertEqual(title_positions, field_positions)

            for header in headers:
                buttons = {
                    button.objectName(): button
                    for button in header.findChildren(QPushButton)
                }
                run_button = buttons["saveRunButton"]
                export_button = buttons["exportSettingButton"]
                button_row = run_button.parentWidget()
                self.assertEqual(run_button.height(), export_button.height())
                self.assertEqual(run_button.height(), button_row.height())
                self.assertEqual(button_row.width(), title_start_x)
                self.assertEqual(button_row.layout().spacing(), 10)
                self.assertEqual(
                    button_row.layout().direction(),
                    QBoxLayout.Direction.LeftToRight,
                )
                full_button_width = (
                    button_row.width() - button_row.layout().spacing()
                ) // 2
                self.assertEqual(
                    run_button.width(),
                    full_button_width,
                )
                self.assertEqual(
                    export_button.width(),
                    round(full_button_width * 0.6),
                )
                self.assertEqual(
                    run_button.width()
                    + export_button.width()
                    + button_row.layout().spacing()
                    + button_row.layout().contentsMargins().right(),
                    button_row.width(),
                )
                self.assertGreater(
                    button_row.layout().contentsMargins().right(),
                    0,
                )
                self.assertEqual(export_button.text(), "Export\nSetting")
                self.assertEqual(
                    export_button.accessibleName(),
                    "Export Settings",
                )
                self.assertIn("shared settings file", run_button.toolTip())
                self.assertIn("standalone JSON file", export_button.toolTip())
                self.assertNotIn("\n", run_button.text())
                export_lines = export_button.text().splitlines()
                self.assertEqual(export_lines, ["Export", "Setting"])
                self.assertLessEqual(
                    max(
                        export_button.fontMetrics().horizontalAdvance(line)
                        for line in export_lines
                    )
                    + 16,
                    export_button.width(),
                )
                self.assertLessEqual(
                    (export_button.fontMetrics().height() * len(export_lines))
                    + 2,
                    export_button.height(),
                )
        finally:
            for card in cards:
                card.close()

    def test_directory_save_button_is_one_and_a_half_times_wide_and_left_aligned(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtWidgets import (
            QFormLayout,
            QFrame,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QWidget,
        )

        card = QFrame()
        layout = QFormLayout(card)
        layout.setHorizontalSpacing(30)

        header = QWidget()
        header.setObjectName("toolHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        actions = QWidget()
        actions.setObjectName("directoryActionButtons")
        actions.setProperty("originalSingleButtonHeight", 40)
        action_layout = QHBoxLayout(actions)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(0)

        save_button = QPushButton("Save Directories")
        save_button.setObjectName("saveDirectoriesButton")
        save_button.setStyleSheet("font-weight: bold; padding: 10px 16px;")
        action_layout.addWidget(save_button)
        action_layout.addStretch()

        title = QLabel("Global Directory Settings")
        title.setObjectName("toolTitle")
        header_layout.addWidget(
            actions,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        header_layout.addWidget(title, 1)
        field = QLineEdit()
        layout.addRow(header)
        layout.addRow(QLabel("Alignment Report Directory:"), field)

        shared_label_width = self.tools_gui_class._align_form_label_columns([layout])
        title_start_x = self.tools_gui_class._align_tool_card_headers(
            [layout],
            shared_label_width,
        )
        former_width = max(1, (title_start_x - 10) // 2)

        try:
            card.resize(1000, 100)
            card.show()
            self.app.processEvents()

            self.assertEqual(save_button.width(), round(former_width * 1.5))
            self.assertEqual(
                save_button.mapTo(card, QPoint(0, 0)).x(),
                actions.mapTo(card, QPoint(0, 0)).x(),
            )
            self.assertEqual(
                title.mapTo(card, QPoint(0, 0)).x(),
                field.mapTo(card, QPoint(0, 0)).x(),
            )
            self.assertGreaterEqual(
                save_button.width(),
                save_button.fontMetrics().horizontalAdvance(save_button.text()) + 32,
            )
        finally:
            card.close()

    def test_legacy_path_directory_does_not_restore_a_directory_row(self):
        from PySide6.QtWidgets import QLabel, QTabWidget, QWidget

        fake_window = QWidget()
        fake_window.save_directories = lambda: None
        fake_window.tip_db = {}
        fake_window._tool_form_layouts = []
        fake_window.tabs = QTabWidget()
        fake_window.tab_paths = []
        legacy_settings = {
            "DIRECTORIES": {
                "FASTA_DIR": "custom_sequences",
                "PATH_DIR": "legacy_paths",
            }
        }

        with mock.patch("os.path.exists", return_value=True), mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps(legacy_settings)),
        ):
            self.tools_gui_class.create_directories_tab(fake_window)

        try:
            labels = {
                label.text()
                for label in fake_window.tabs.findChildren(QLabel)
            }
            self.assertNotIn("PATH_DIR", fake_window.dir_inputs)
            self.assertNotIn("Alignment Path Directory:", labels)
            self.assertEqual(
                fake_window.dir_inputs["FASTA_DIR"].text(),
                "custom_sequences",
            )
        finally:
            fake_window.close()

    def test_directory_open_buttons_precede_browse_and_open_selected_folder(self):
        from PySide6.QtWidgets import QTabWidget, QWidget

        fake_window = QWidget()
        fake_window.save_directories = lambda: None
        fake_window.tip_db = {}
        fake_window._tool_form_layouts = []
        fake_window.tabs = QTabWidget()
        fake_window.tab_paths = []
        self.tools_gui_class.create_directories_tab(fake_window)

        try:
            self.assertEqual(
                set(fake_window.directory_open_buttons),
                set(DEFAULT_DIRECTORY_PATHS),
            )
            for key, button in fake_window.directory_open_buttons.items():
                with self.subTest(key=key):
                    row_layout = button.parentWidget().layout()
                    widgets = [
                        row_layout.itemAt(index).widget()
                        for index in range(row_layout.count())
                    ]
                    button_index = widgets.index(button)
                    self.assertIs(widgets[button_index - 1], fake_window.dir_inputs[key])
                    self.assertEqual(widgets[button_index + 1].text(), "Browse...")

            with tempfile.TemporaryDirectory() as temp_dir:
                selected_folder = pathlib.Path(temp_dir, "selected", "embeddings")
                fake_window.dir_inputs["EMBED_DIR"].setText(str(selected_folder))
                with mock.patch(
                    "PySide6.QtGui.QDesktopServices.openUrl", return_value=True
                ) as open_url:
                    fake_window.directory_open_buttons["EMBED_DIR"].click()

                self.assertTrue(selected_folder.is_dir())
                self.assertEqual(
                    pathlib.Path(open_url.call_args.args[0].toLocalFile()).resolve(),
                    selected_folder.resolve(),
                )
        finally:
            fake_window.close()

    def test_export_writes_current_values_and_only_required_directories(self):
        from PySide6.QtWidgets import QCheckBox, QLineEdit, QMessageBox, QInputDialog

        script_path = str(SRC_DIR / "tools" / "Sanitize_Sequences.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            shared_settings = pathlib.Path(temp_dir) / "shared.json"
            shared_settings.write_text('{"sentinel": true}', encoding="utf-8")

            input_line = QLineEdit("current.fasta")
            overwrite = QCheckBox()
            overwrite.setChecked(True)
            fake_window = SimpleNamespace()
            fake_window.dir_inputs = {
                "FASTA_DIR": QLineEdit(r"current_sequences\nested"),
                "EMBED_DIR": QLineEdit("should_not_export"),
                "SETTING_EXPORT_DIR": QLineEdit(temp_dir),
            }
            fake_window.script_data = {
                script_path: {
                    "inputs": {
                        "INPUT_FASTA": {"widget": input_line, "type": "text"},
                        "OVER_WRITE": {"widget": overwrite, "type": "switch"},
                    },
                    "settings": [
                        {"name": "INPUT_FASTA"},
                        {"name": "OVER_WRITE"},
                    ],
                }
            }
            fake_window._normalized_export_filename = (
                self.tools_gui_class._normalized_export_filename
            )
            fake_window._current_directory_settings = lambda: (
                self.tools_gui_class._current_directory_settings(fake_window)
            )
            fake_window._portable_export_directory_path = (
                self.tools_gui_class._portable_export_directory_path
            )
            fake_window._collect_tool_settings = lambda path: (
                self.tools_gui_class._collect_tool_settings(fake_window, path)
            )

            with mock.patch.object(
                QInputDialog, "getText", return_value=("portable", True)
            ), mock.patch.object(QMessageBox, "information") as information, mock.patch.object(
                QMessageBox, "critical"
            ) as critical:
                self.tools_gui_class.export_settings(fake_window, script_path)

            payload = json.loads(
                (pathlib.Path(temp_dir) / "portable.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["DIRECTORIES"],
                {"FASTA_DIR": "current_sequences/nested"},
            )
            self.assertEqual(
                payload["Sanitize_Sequences.py"],
                {"INPUT_FASTA": "current.fasta", "OVER_WRITE": True},
            )
            self.assertEqual(
                shared_settings.read_text(encoding="utf-8"), '{"sentinel": true}'
            )
            information.assert_called_once()
            critical.assert_not_called()
            self.assertEqual(list(pathlib.Path(temp_dir).glob("*.partial")), [])

            exported_path = pathlib.Path(temp_dir) / "portable.json"
            original_export = exported_path.read_text(encoding="utf-8")
            input_line.setText("changed.fasta")
            with mock.patch.object(
                QInputDialog, "getText", return_value=("portable", True)
            ), mock.patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.StandardButton.No,
            ) as question:
                self.tools_gui_class.export_settings(fake_window, script_path)
            question.assert_called_once()
            self.assertEqual(
                exported_path.read_text(encoding="utf-8"), original_export
            )

            with mock.patch.object(
                QInputDialog, "getText", return_value=("cancelled", False)
            ):
                self.tools_gui_class.export_settings(fake_window, script_path)
            self.assertFalse((pathlib.Path(temp_dir) / "cancelled.json").exists())


if __name__ == "__main__":
    unittest.main()
