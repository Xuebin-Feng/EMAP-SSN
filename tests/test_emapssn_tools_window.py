# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tools window (EMAPSSN_Tools.ToolsGUI).

Tooltip routing, BLAST custom-column controls, tool cards and their headers,
the Directories tab, settings export, hardware-dependent precision and
execution options, the host-cache control, the model dropdown and the file
dropdowns.
"""

import ast
import json
import os
import pathlib
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint  # noqa: E402
from PySide6.QtGui import QHelpEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QFormLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from EMAPSSN_Tools import (  # noqa: E402
    SpacedTipLabel,
    ToolsGUI,
    _configure_linux_qtwebengine_rendering,
    bind_custom_blast_column_controls,
)
from tests.tools_gui_fixtures import isolated_tools_project  # noqa: E402
from tools.tool_helpers.Tool_Pipeline import DEFAULT_DIRECTORY_PATHS  # noqa: E402


class TooltipRoutingWindow(ToolsGUI):
    def __init__(self):
        QMainWindow.__init__(self)
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        self.native_tip_button = QPushButton("Native tooltip", central_widget)
        self.native_tip_button.setToolTip("Help from the widget tooltip.")
        self.shared_tip_label = QLabel("Shared tooltip", central_widget)
        layout.addWidget(self.native_tip_button)
        layout.addWidget(self.shared_tip_label)
        self.setCentralWidget(central_widget)

        self.tip_panel = SpacedTipLabel("Initial help")
        self.tip_db = {self.shared_tip_label: "Help from the shared database."}
        self.shared_tip_label.installEventFilter(self)
        self._route_native_tooltips_to_tip_panel()


class TooltipRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = TooltipRoutingWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_linux_webengine_software_rendering_preserves_existing_flags(self):
        environment = {"QTWEBENGINE_CHROMIUM_FLAGS": "--disable-logging"}

        changed = _configure_linux_qtwebengine_rendering(environment, "linux")

        self.assertTrue(changed)
        self.assertEqual(
            environment["QTWEBENGINE_CHROMIUM_FLAGS"],
            "--disable-logging --disable-gpu",
        )
        self.assertFalse(
            _configure_linux_qtwebengine_rendering(environment, "linux")
        )

    def test_webengine_rendering_is_unchanged_outside_linux(self):
        for platform_name in ("darwin", "win32"):
            with self.subTest(platform_name=platform_name):
                environment = {
                    "QTWEBENGINE_CHROMIUM_FLAGS": "--disable-logging"
                }

                changed = _configure_linux_qtwebengine_rendering(
                    environment, platform_name
                )

                self.assertFalse(changed)
                self.assertEqual(
                    environment["QTWEBENGINE_CHROMIUM_FLAGS"],
                    "--disable-logging",
                )

    def test_native_tooltip_event_updates_bottom_panel_and_is_suppressed(self):
        event = QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), QPoint(1, 1))

        QApplication.sendEvent(self.window.native_tip_button, event)

        self.assertIn(
            "Help from the widget tooltip.",
            self.window.tip_panel.text(),
        )
        self.assertTrue(
            self.window.eventFilter(
                self.window.native_tip_button,
                QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), QPoint(1, 1)),
            )
        )

    def test_shared_database_tip_still_updates_bottom_panel(self):
        event = QEvent(QEvent.Type.Enter)

        handled = self.window.eventFilter(self.window.shared_tip_label, event)

        self.assertFalse(handled)
        self.assertIn(
            "Help from the shared database.",
            self.window.tip_panel.text(),
        )

    def test_custom_blast_columns_follow_layout_selection(self):
        layout_combo = QComboBox()
        layout_combo.addItem("standard_outfmt6", "standard_outfmt6")
        layout_combo.addItem("outfmt7_fields", "outfmt7_fields")
        layout_combo.addItem(
            "Custom Columns (1-based indexing)", "custom_columns"
        )
        layout_combo.setProperty("persistItemData", True)
        inputs = {"BLAST_LAYOUT": {"widget": layout_combo}}
        row_widgets = {}
        for name in ("QUERY_COLUMN", "SUBJECT_COLUMN", "EVALUE_COLUMN"):
            label = QLabel(name)
            widget = QSpinBox()
            inputs[name] = {"widget": widget}
            row_widgets[name] = (label, widget)

        bind_custom_blast_column_controls(inputs, row_widgets)

        self.assertTrue(
            all(not inputs[name]["widget"].isEnabled() for name in row_widgets)
        )
        self.assertTrue(all(not label.isEnabled() for label, _ in row_widgets.values()))
        layout_combo.setCurrentText("Custom Columns (1-based indexing)")
        self.app.processEvents()
        self.assertEqual(layout_combo.currentData(), "custom_columns")
        self.assertTrue(
            all(inputs[name]["widget"].isEnabled() for name in row_widgets)
        )
        self.assertTrue(all(label.isEnabled() for label, _ in row_widgets.values()))

        self.window.script_data = {
            "parse": {
                "inputs": {
                    "BLAST_LAYOUT": {
                        "widget": layout_combo,
                        "type": "dropdown",
                    }
                },
                "settings": [{"name": "BLAST_LAYOUT", "def": {}}],
            }
        }
        self.assertEqual(
            self.window._collect_tool_settings("parse")["BLAST_LAYOUT"],
            "custom_columns",
        )

    def test_custom_blast_column_controls_are_merged_into_one_row(self):
        container = QWidget()
        form = QFormLayout(container)
        row_widgets = {}
        for name in ("QUERY_COLUMN", "SUBJECT_COLUMN", "EVALUE_COLUMN"):
            label = QLabel(name)
            widget = QSpinBox()
            form.addRow(label, widget)
            row_widgets[name] = (label, widget)

        ToolsGUI._merge_inline_field_rows(
            form, "Parse_BLAST_Output.py", row_widgets
        )

        self.assertEqual(form.rowCount(), 1)
        field = form.itemAt(0, QFormLayout.ItemRole.SpanningRole).widget()
        self.assertEqual(field.property("compactColumnRatio"), "1:1:1")


class ToolExportGuiTests(unittest.TestCase):
    def test_numeric_text_export_is_typed_without_changing_execution_collection(self):
        from PySide6.QtWidgets import QLineEdit, QMessageBox, QInputDialog
        from mcp_server.pipeline.Pipeline_Settings import normalize_pipeline_settings

        script_path = str(SRC_DIR / "tools" / "Align_Similarity_Matrix.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            fields = {"INPUT_HDF5": QLineEdit("example.h5"), "BATCH_SIZE": QLineEdit("12345")}
            window = SimpleNamespace(
                dir_inputs={"SETTING_EXPORT_DIR": QLineEdit(temp_dir)},
                script_data={script_path: {
                    "inputs": {key: {"widget": widget, "type": "text"} for key, widget in fields.items()},
                    "settings": [{"name": key} for key in fields],
                }},
            )
            window._normalized_export_filename = self.tools_gui_class._normalized_export_filename
            window._current_directory_settings = lambda: self.tools_gui_class._current_directory_settings(window)
            window._collect_tool_settings = lambda path: self.tools_gui_class._collect_tool_settings(window, path)
            before = window._collect_tool_settings(script_path)
            with mock.patch.object(QInputDialog, "getText", return_value=("typed", True)), \
                    mock.patch.object(QMessageBox, "information"), \
                    mock.patch.object(QMessageBox, "critical") as critical:
                self.tools_gui_class.export_settings(window, script_path)
                critical.assert_not_called()
            exported_path = pathlib.Path(temp_dir) / "typed.json"
            document = json.loads(exported_path.read_text())
            self.assertEqual(document["Align_Similarity_Matrix.py"]["BATCH_SIZE"], 12345)
            self.assertTrue(normalize_pipeline_settings(
                "align_similarity_matrix", PROJECT_ROOT, settings_document=document
            )["valid"])
            self.assertEqual(window._collect_tool_settings(script_path), before)
            self.assertEqual(before["BATCH_SIZE"], "12345")
            fields["BATCH_SIZE"].setText("not a number")
            with mock.patch.object(QInputDialog, "getText", return_value=("invalid", True)), \
                    mock.patch.object(QMessageBox, "critical") as critical:
                self.tools_gui_class.export_settings(window, script_path)
                critical.assert_called_once()
                self.assertIn("BATCH_SIZE", critical.call_args.args[-1])
            self.assertFalse((pathlib.Path(temp_dir) / "invalid.json").exists())

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

    def test_execution_mode_gui_contract_and_export_round_trip(self):
        from PySide6.QtWidgets import QComboBox, QInputDialog, QLineEdit, QMessageBox

        source = (SRC_DIR / "EMAPSSN_Tools.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count('"var_name": "EXECUTION_MODE"'), 2)
        self.assertGreaterEqual(
            source.count('"option_values": ["auto", "scalar", "tiled"]'), 2
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
        from utilities import Hardware_Acceleration as Hardware_Utils
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

        # Without these, the BF16 half of the sync runs a real BF16 probe on cuda:0.
        with mock.patch(
            "EMAPSSN_Tools.is_nvidia_cuda",
            side_effect=lambda selected: selected.type == "cuda",
        ), mock.patch(
            "EMAPSSN_Tools.bf16_accelerator_support",
            side_effect=lambda selected: (
                selected.type == "cuda",
                "mock capability",
            ),
        ), mock.patch.object(
            Hardware_Utils, "get_available_devices", return_value=[cpu]
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
        from utilities import Hardware_Acceleration as Hardware_Utils
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
        from desktop.Desktop_App import add_combo_options
        from utilities import Hardware_Acceleration as Hardware_Utils
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
        # Labels differ from the stored values, as they will once translated.
        add_combo_options(
            execution, ["auto", "scalar", "tiled"], ["Auto", "Scalar", "Tiled"]
        )

        with mock.patch(
            "EMAPSSN_Tools.tiled_accelerator_support",
            return_value=(True, "mock support"),
        ):
            execution.setCurrentIndex(execution.findData("tiled"))
            self.assertFalse(
                self.sync_alignment_tiled_option(
                    device, execution, [cpu, mps]
                )
            )
            self.assertEqual(execution.currentData(), "auto")
            self.assertEqual(execution.findData("tiled"), -1)
            self.assertFalse(execution.property("tiledAvailable"))

            self.assertTrue(
                self.sync_alignment_tiled_option(
                    device, execution, [cpu, mps, xpu]
                )
            )
            self.assertGreaterEqual(execution.findData("tiled"), 0)

            device.setCurrentIndex(device.findData("mps"))
            self.assertFalse(
                self.sync_alignment_tiled_option(
                    device, execution, [cpu, mps, xpu]
                )
            )
            self.assertEqual(execution.findData("tiled"), -1)

            device.setCurrentIndex(device.findData("xpu:0"))
            self.assertTrue(
                self.sync_alignment_tiled_option(
                    device,
                    execution,
                    [cpu, mps, xpu],
                    allow_mps=True,
                )
            )
            self.assertGreaterEqual(execution.findData("tiled"), 0)

            device.setCurrentIndex(device.findData("mps"))
            self.assertTrue(
                self.sync_alignment_tiled_option(
                    device,
                    execution,
                    [cpu, mps, xpu],
                    allow_mps=True,
                )
            )
            self.assertGreaterEqual(execution.findData("tiled"), 0)

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

    def test_embedding_row_widens_model_name_at_the_cost_of_saving_mode(self):
        from PySide6.QtWidgets import (
            QComboBox,
            QFormLayout,
            QLabel,
            QSizePolicy,
            QWidget,
        )

        def build_row():
            form_parent = QWidget()
            layout = QFormLayout(form_parent)
            layout.setHorizontalSpacing(30)
            row_widgets = {}
            definitions = (
                ("MODEL_NAME", "Model Name:",
                 ["ankh_large [non-commercial]", "esm2_t33_650M_UR50D"]),
                ("SAVING_MODE", "Saving Mode:", ["float32", "float16"]),
                ("DEVICE_SELECTION", "Device:", ["Auto Benchmark", "CPU"]),
            )
            for var_name, label_text, options in definitions:
                label = QLabel(label_text)
                field = QComboBox()
                field.addItems(options)
                field.setMinimumContentsLength(12)
                field.setSizeAdjustPolicy(
                    QComboBox.SizeAdjustPolicy
                    .AdjustToMinimumContentsLengthWithIcon
                )
                field.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
                )
                layout.addRow(label, field)
                row_widgets[var_name] = (label, field)

            self.tools_gui_class._merge_compact_rows(
                layout, "Generate_Embeddings.py", row_widgets
            )
            # The leading label shares the form's label column width.
            row_widgets["MODEL_NAME"][0].setFixedWidth(
                QLabel("Sequence Set (.fasta):").sizeHint().width()
            )
            form_parent.show()
            self.app.processEvents()
            return form_parent, layout, row_widgets

        def field_widths(width):
            form_parent, layout, row_widgets = build_row()
            form_parent.resize(width, form_parent.sizeHint().height())
            layout.activate()
            self.app.processEvents()
            compact = form_parent.findChild(
                QWidget, "compactRow_MODEL_NAME_SAVING_MODE_DEVICE_SELECTION"
            )
            self.assertIsNotNone(compact)
            self.assertFalse(compact.property("stacked"))
            widths = {
                name: field.width() for name, (_, field) in row_widgets.items()
            }
            form_parent.close()
            return widths

        form_parent = build_row()[0]
        compact = form_parent.findChild(
            QWidget, "compactRow_MODEL_NAME_SAVING_MODE_DEVICE_SELECTION"
        )
        self.assertEqual(compact.property("compactColumnRatio"), "5:3:4")
        form_parent.close()

        widths = field_widths(1700)
        self.assertGreater(widths["MODEL_NAME"], widths["SAVING_MODE"])
        # Saving mode only ever shows "float32"/"float16"; it keeps a legible
        # dropdown while the surrendered space goes to the model names.
        self.assertGreaterEqual(
            widths["SAVING_MODE"],
            QComboBox().fontMetrics().horizontalAdvance("float32") * 2,
        )
        # The device column keeps the third of the row it had before.
        self.assertLess(
            abs(widths["DEVICE_SELECTION"] - widths["MODEL_NAME"]), 100
        )

    def test_tab_pages_release_shared_content_width_without_resizing_tab_labels(self):
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
        self.tools_gui_class._harmonize_tab_page_widths(fake_window)

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
            self.assertEqual(content_page.minimumWidth(), 0)

    def test_tool_cards_split_into_a_left_and_a_right_section(self):
        """A card's left section holds its action buttons and field labels and is
        as wide as the widest such element across all cards; the title and the
        fields start one horizontal spacing after it."""
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QFrame, QLabel, QLineEdit, QPushButton
        from desktop.Desktop_App import (
            BUTTON_TEXT_PADDING, clipped_button_text, text_button_width,
        )

        fake_window = SimpleNamespace(
            tool_titles={},
            save_and_run=lambda script_path: None,
            export_settings=lambda script_path: None,
        )

        def natural_width(text):
            return QLabel(text).sizeHint().width()

        # A long label widens the section; with short labels the buttons do.
        for label_texts in (("Short:", "Normalized Noise Scale (0 to 0.1):"),
                            ("A:", "B:")):
            with self.subTest(labels=label_texts):
                cards, layouts, fields, headers = [], [], [], []
                for label_text in label_texts:
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

                left_width, title_start_x = self.tools_gui_class._align_tool_cards(layouts)
                try:
                    for card in cards:
                        card.resize(1000, 100)
                        card.show()
                    self.app.processEvents()

                    buttons = {button.objectName(): button
                               for button in headers[0].findChildren(QPushButton)}
                    run_button = buttons["saveRunButton"]
                    export_button = buttons["exportSettingButton"]
                    action_width = run_button.width() + 10 + export_button.width()
                    widest = max([action_width]
                                 + [natural_width(text) for text in label_texts])
                    self.assertEqual(left_width, widest)
                    self.assertEqual(title_start_x, left_width + 30)

                    field_positions = {
                        field.mapTo(card, QPoint(0, 0)).x()
                        for card, field in zip(cards, fields)
                    }
                    title_positions = {
                        header.findChild(QLabel, "toolTitle").mapTo(card, QPoint(0, 0)).x()
                        for card, header in zip(cards, headers)
                    }
                    self.assertEqual(len(field_positions), 1)
                    self.assertEqual(title_positions, field_positions)

                    for header in headers:
                        buttons = {button.objectName(): button
                                   for button in header.findChildren(QPushButton)}
                        run_button = buttons["saveRunButton"]
                        export_button = buttons["exportSettingButton"]
                        button_row = run_button.parentWidget()
                        self.assertEqual(button_row.width(), title_start_x)
                        self.assertEqual(run_button.x(), 0)
                        self.assertEqual(
                            export_button.x(), run_button.width() + button_row.layout().spacing()
                        )
                        for button in (run_button, export_button):
                            self.assertEqual(
                                button.width(),
                                text_button_width([button.text()], button.font()),
                            )
                            self.assertTrue(button.font().bold())
                            self.assertEqual(clipped_button_text(button), 0)
                        self.assertEqual(run_button.height(), export_button.height())
                        self.assertEqual(run_button.font().pointSizeF(),
                                         export_button.font().pointSizeF())
                        # "Save && Run" shows as "Save & Run".
                        self.assertEqual(
                            run_button.width()
                            - run_button.fontMetrics().horizontalAdvance("Save & Run"),
                            2 * BUTTON_TEXT_PADDING,
                        )
                        self.assertEqual(export_button.text(), "Export")
                        self.assertEqual(export_button.accessibleName(), "Export Settings")
                        self.assertIn("shared settings file", run_button.toolTip())
                        for phrase in ("standalone JSON file", "Setting Export Directory",
                                       "does not run"):
                            self.assertIn(phrase, export_button.toolTip())
                finally:
                    for card in cards:
                        card.close()

    def test_directory_save_button_fits_its_text_and_is_left_aligned(self):
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QLabel, QPushButton, QTabWidget, QWidget
        from desktop.Desktop_App import BUTTON_TEXT_PADDING, clipped_button_text

        isolated_tools_project(self)
        fake_window = QWidget()
        fake_window.save_directories = lambda: None
        fake_window.tip_db = {}
        fake_window._tool_form_layouts = []
        fake_window.tabs = QTabWidget()
        fake_window.tab_paths = []
        self.tools_gui_class.create_directories_tab(fake_window)

        [layout] = fake_window._tool_form_layouts
        card = layout.parentWidget()
        save_button = card.findChild(QPushButton, "saveDirectoriesButton")
        actions = card.findChild(QWidget, "directoryActionButtons")
        title = card.findChild(QLabel, "toolTitle")
        # The "Alignment Report Directory:" row.
        field = fake_window.dir_inputs["REPORT_DIR"]

        _left_width, title_start_x = self.tools_gui_class._align_tool_cards([layout])

        try:
            fake_window.tabs.resize(1100, 600)
            fake_window.tabs.show()
            self.app.processEvents()

            self.assertEqual(
                save_button.width()
                - save_button.fontMetrics().horizontalAdvance(save_button.text()),
                2 * BUTTON_TEXT_PADDING,
            )
            self.assertEqual(clipped_button_text(save_button), 0)
            self.assertEqual(actions.width(), title_start_x)
            self.assertEqual(
                save_button.mapTo(card, QPoint(0, 0)).x(),
                actions.mapTo(card, QPoint(0, 0)).x(),
            )
            self.assertEqual(
                title.mapTo(card, QPoint(0, 0)).x(),
                field.mapTo(card, QPoint(0, 0)).x(),
            )
        finally:
            fake_window.tabs.close()
            fake_window.close()

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

        isolated_tools_project(self)
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
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "EMAPSSN_Tools._PROJECT_ROOT", temp_dir
        ):
            # The shared settings file, which an export must leave alone.
            shared_settings = pathlib.Path(temp_dir) / "tools_settings.json"
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


class ToolsModelDropdownTests(unittest.TestCase):
    def open_tools(self, saved_model):
        """Open the Tools window with MODEL_NAME saved in a scratch project root."""
        from PySide6.QtWidgets import QApplication, QTextBrowser
        import EMAPSSN_Tools

        app = QApplication.instance() or QApplication([])
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        settings = pathlib.Path(root.name, "tools_settings.json")
        settings.write_text(
            json.dumps({"Generate_Embeddings.py": {"MODEL_NAME": saved_model}}),
            encoding="utf-8",
        )
        for patcher in (
            mock.patch.object(EMAPSSN_Tools, "_PROJECT_ROOT", root.name),
            mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        window = EMAPSSN_Tools.ToolsGUI()

        def close():
            window.close()
            window.deleteLater()
            app.processEvents()

        self.addCleanup(close)
        path, data = next(
            (key, value) for key, value in window.script_data.items()
            if pathlib.Path(key).name == "Generate_Embeddings.py"
        )
        return EMAPSSN_Tools, window, path, data["inputs"]["MODEL_NAME"]["widget"], settings

    def test_removed_saved_model_stays_visible_and_blocks_running(self):
        tools, window, path, combo, settings = self.open_tools("esm2_t36_3b")
        self.assertEqual(combo.currentData(), "esm2_t36_3b")
        self.assertEqual(combo.currentText(), "Unavailable saved model [esm2_t36_3b]")
        before = settings.read_text(encoding="utf-8")

        with mock.patch.object(tools.QMessageBox, "critical") as critical:
            window.save_and_run(path)

        critical.assert_called_once()
        self.assertIn("'esm2_t36_3b' is no longer supported", critical.call_args.args[2])
        self.assertEqual(settings.read_text(encoding="utf-8"), before)

    def test_supported_saved_model_is_selected_normally(self):
        _, _, _, combo, _ = self.open_tools("esm2_t6_8m")
        self.assertEqual(combo.currentData(), "esm2_t6_8m")
        self.assertEqual(combo.findText("Unavailable saved model [esm2_t6_8m]"), -1)
        self.assertFalse(combo.currentText().startswith("Unavailable"))


class SharedSettingsSaveTests(unittest.TestCase):
    """Save Directories and a tool's Run keep a tools_settings.json they cannot load."""

    UNREADABLE = (
        '{"Generate_Embeddings.py": {"MODEL_NAME": "esm2_t6_8m"},}',
        "[1, 2]",
    )
    SANITIZE = str(SRC_DIR / "tools" / "Sanitize_Sequences.py")

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def save_directories(self, root):
        """Run Save Directories with FASTA_DIR set; return (critical, information)."""
        import EMAPSSN_Tools
        from PySide6.QtWidgets import QLineEdit

        window = SimpleNamespace(dir_inputs={
            "FASTA_DIR": QLineEdit(str(root / "sequences")),
            "EMBED_DIR": QLineEdit("  "),
        })
        with mock.patch.object(EMAPSSN_Tools.QMessageBox, "critical") as critical, \
                mock.patch.object(EMAPSSN_Tools.QMessageBox, "information") as information:
            ToolsGUI.save_directories(window)
        return critical, information

    def run_sanitize(self):
        """Run Sanitize_Sequences.py from the GUI; return (critical, information, launch)."""
        import EMAPSSN_Tools

        window = SimpleNamespace(_collect_tool_settings=lambda path: {"INPUT_FASTA": "new.fasta"})
        with mock.patch.object(EMAPSSN_Tools.QMessageBox, "critical") as critical, \
                mock.patch.object(EMAPSSN_Tools.QMessageBox, "information") as information, \
                mock.patch.object(EMAPSSN_Tools, "launch_in_terminal") as launch, \
                mock.patch("builtins.print"):
            ToolsGUI.save_and_run(window, self.SANITIZE)
        return critical, information, launch

    def test_directory_save_replaces_only_the_directories(self):
        root = isolated_tools_project(
            self, {"Generate_Embeddings.py": {"MODEL_NAME": "esm2_t6_8m"}}
        )
        critical, information = self.save_directories(root)

        critical.assert_not_called()
        information.assert_called_once()
        document = json.loads((root / "tools_settings.json").read_text(encoding="utf-8"))
        self.assertEqual(document, {
            "DIRECTORIES": {
                "FASTA_DIR": os.path.normpath(str(root / "sequences")),
                "EMBED_DIR": "",
            },
            "Generate_Embeddings.py": {"MODEL_NAME": "esm2_t6_8m"},
        })
        self.assertEqual(os.listdir(root), ["tools_settings.json"])

    def test_directory_save_keeps_a_file_it_cannot_load(self):
        root = isolated_tools_project(self)
        settings = root / "tools_settings.json"
        for content in self.UNREADABLE:
            with self.subTest(content=content):
                settings.write_text(content, encoding="utf-8")
                critical, information = self.save_directories(root)

                self.assertEqual(settings.read_text(encoding="utf-8"), content)
                information.assert_not_called()
                critical.assert_called_once()
                message = critical.call_args.args[2]
                self.assertIn(str(settings), message)
                self.assertIn("left unchanged", message)

    def test_directory_save_that_fails_midway_keeps_the_previous_file(self):
        root = isolated_tools_project(
            self, {"Generate_Embeddings.py": {"MODEL_NAME": "esm2_t6_8m"}}
        )
        settings = root / "tools_settings.json"
        before = settings.read_bytes()

        def fail_midway(document, handle, **options):
            handle.write('{"DIRECTORIES": ')
            raise OSError(28, "No space left on device")

        with mock.patch.object(json, "dump", side_effect=fail_midway):
            critical, information = self.save_directories(root)

        information.assert_not_called()
        self.assertIn("No space left on device", critical.call_args.args[2])
        self.assertEqual(settings.read_bytes(), before)
        self.assertEqual(os.listdir(root), ["tools_settings.json"])

    def test_run_saves_its_section_and_keeps_the_others(self):
        root = isolated_tools_project(
            self, {"Generate_Embeddings.py": {"MODEL_NAME": "esm2_t6_8m"}}
        )
        critical, information, launch = self.run_sanitize()

        critical.assert_not_called()
        launch.assert_called_once()
        information.assert_called_once()
        document = json.loads((root / "tools_settings.json").read_text(encoding="utf-8"))
        self.assertEqual(document["Sanitize_Sequences.py"], {"INPUT_FASTA": "new.fasta"})
        self.assertEqual(document["Generate_Embeddings.py"], {"MODEL_NAME": "esm2_t6_8m"})

    def test_run_keeps_a_file_it_cannot_load_and_starts_nothing(self):
        root = isolated_tools_project(self)
        settings = root / "tools_settings.json"
        for content in self.UNREADABLE:
            with self.subTest(content=content):
                settings.write_text(content, encoding="utf-8")
                critical, information, launch = self.run_sanitize()

                self.assertEqual(settings.read_text(encoding="utf-8"), content)
                launch.assert_not_called()
                information.assert_not_called()
                critical.assert_called_once()
                message = critical.call_args.args[2]
                self.assertIn(str(settings), message)
                self.assertIn("not started", message)


class SettingsLoadReportTests(unittest.TestCase):
    """Opening the window reports a tools_settings.json it cannot load, and keeps it."""

    def open_tools(self, content):
        """Open the Tools window over a scratch project whose settings file holds content."""
        import EMAPSSN_Tools
        from PySide6.QtWidgets import QTextBrowser

        app = QApplication.instance() or QApplication([])
        settings = isolated_tools_project(self) / "tools_settings.json"
        settings.write_text(content, encoding="utf-8")
        patcher = mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser)
        patcher.start()
        self.addCleanup(patcher.stop)
        window = EMAPSSN_Tools.ToolsGUI()

        def close():
            window.close()
            window.deleteLater()
            app.processEvents()

        self.addCleanup(close)
        return EMAPSSN_Tools, window, settings

    def test_an_unreadable_file_is_reported_when_the_window_opens(self):
        for content in SharedSettingsSaveTests.UNREADABLE:
            with self.subTest(content=content):
                tools, window, settings = self.open_tools(content)
                self.assertEqual(
                    window.dir_inputs["FASTA_DIR"].text(), DEFAULT_DIRECTORY_PATHS["FASTA_DIR"]
                )
                with mock.patch.object(tools.QMessageBox, "warning") as warning:
                    window.report_settings_load_error()

                warning.assert_called_once()
                message = warning.call_args.args[2]
                self.assertIn(str(settings), message)
                self.assertIn("left unchanged", message)
                self.assertEqual(settings.read_text(encoding="utf-8"), content)

    def test_saved_values_are_shown_and_nothing_is_reported(self):
        tools, window, _ = self.open_tools(json.dumps({
            "DIRECTORIES": {"FASTA_DIR": "custom_sequences", "MSA_DIR": ""},
            "Generate_Embeddings.py": {"SAVING_MODE": "float16"},
        }))
        self.assertEqual(window.dir_inputs["FASTA_DIR"].text(), "custom_sequences")
        # A path saved blank stays blank rather than showing the default.
        self.assertEqual(window.dir_inputs["MSA_DIR"].text(), "")
        saving_mode = next(
            data["inputs"]["SAVING_MODE"]["widget"] for path, data in window.script_data.items()
            if pathlib.Path(path).name == "Generate_Embeddings.py"
        )
        self.assertEqual(saving_mode.currentText(), "float16")
        with mock.patch.object(tools.QMessageBox, "warning") as warning:
            window.report_settings_load_error()
        warning.assert_not_called()

    def test_the_launched_window_reports_once_it_is_shown(self):
        tree = ast.parse((SRC_DIR / "EMAPSSN_Tools.py").read_text(encoding="utf-8"))
        launch = next(
            node for node in tree.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
            and any(ast.unparse(call) == "ToolsGUI()" for call in ast.walk(node) if isinstance(call, ast.Call))
        )
        lines = {ast.unparse(node): node.lineno for node in ast.walk(launch) if isinstance(node, ast.Call)}
        self.assertLess(lines["show_window_in_front(window)"], lines["window.report_settings_load_error()"])
        self.assertLess(lines["window.report_settings_load_error()"], lines["app.exec()"])


class FolderDropdownTests(unittest.TestCase):
    """A file dropdown lists its directory as the tools find it and starts on the saved file.

    It does so as the window opens, without being opened itself: before the
    Directories tab exists, the directory comes from the saved settings, or in a
    language redraw from the ones carried over. A relative directory is relative to
    the project root, not the working directory, and a blank one means the default.
    """

    # One file dropdown per directory: tool, field, directory key, extension.
    FIELDS = (
        ("Sanitize_Sequences.py", "INPUT_FASTA", "FASTA_DIR", ".fasta"),
        ("Align_Similarity_Matrix.py", "INPUT_HDF5", "EMBED_DIR", ".h5"),
        ("Network_Extraction.py", "INPUT_NET", "NETWORK_DIR", ".h5"),
        ("Sparse_MSA_Converter.py", "INPUT_FASTA", "MSA_DIR", ".fasta"),
    )

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from PySide6.QtWidgets import QTextBrowser

        self.root = isolated_tools_project(self)
        patcher = mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def saved_name(key, extension):
        return f"saved_{key.lower()}{extension}"

    def open_tools(self, directories=None, carried=None):
        """Open the Tools window, saving directories first if they are given.

        directories maps each key to (its saved value, the folder it leads to).
        Each folder holds its field's saved file and a file listed before it.
        """
        import EMAPSSN_Tools

        if directories is not None:
            document = {"DIRECTORIES": {key: saved for key, (saved, _) in directories.items()}}
            for script, field, key, extension in self.FIELDS:
                folder = directories[key][1]
                folder.mkdir(parents=True, exist_ok=True)
                for name in ("a_listed_first" + extension, self.saved_name(key, extension)):
                    (folder / name).touch()
                document[script] = {field: self.saved_name(key, extension)}
            (self.root / "tools_settings.json").write_text(json.dumps(document), encoding="utf-8")
        window = EMAPSSN_Tools.ToolsGUI(carried=carried)

        def close():
            window.close()
            window.deleteLater()
            self.app.processEvents()

        self.addCleanup(close)
        return window

    @staticmethod
    def dropdown(window, script, field):
        """(script path, combo) of a tool's file dropdown."""
        path, data = next(
            (path, data) for path, data in window.script_data.items()
            if pathlib.Path(path).name == script
        )
        return path, data["inputs"][field]["widget"].combo

    def assert_saved_files_selected(self, window):
        for script, field, key, extension in self.FIELDS:
            with self.subTest(script=script, field=field):
                path, combo = self.dropdown(window, script, field)
                saved = self.saved_name(key, extension)
                listed = sorted(combo.itemText(index) for index in range(combo.count()))
                self.assertEqual(listed, sorted(["a_listed_first" + extension, saved]))
                self.assertEqual(combo.currentText(), saved)
                # What Save & Run and Export collect.
                self.assertEqual(window._collect_tool_settings(path)[field], saved)

    def assert_opening_keeps_the_saved_files(self, window):
        """Opening a dropdown lists the Directories tab's directory again."""
        for script, field, _, _ in self.FIELDS:
            self.dropdown(window, script, field)[1].populate()
        self.assert_saved_files_selected(window)

    def test_a_dropdown_starts_on_the_file_saved_in_its_directory(self):
        elsewhere = tempfile.TemporaryDirectory()
        self.addCleanup(elsewhere.cleanup)
        directories = {}
        for _, _, key, _ in self.FIELDS:
            folder = pathlib.Path(elsewhere.name, key.lower())
            directories[key] = (str(folder), folder)

        window = self.open_tools(directories)

        self.assert_saved_files_selected(window)
        self.assert_opening_keeps_the_saved_files(window)

    def test_relative_and_blank_directories_are_found_from_the_project_root(self):
        directories = {}
        for _, _, key, _ in self.FIELDS:
            relative = os.path.join("custom", key.lower())
            directories[key] = (relative, self.root / relative)
        # A blank directory is the default, which the tools also find from the root.
        directories["MSA_DIR"] = ("", self.root / DEFAULT_DIRECTORY_PATHS["MSA_DIR"])
        self.assertNotEqual(pathlib.Path.cwd().resolve(), self.root.resolve())

        window = self.open_tools(directories)

        self.assert_saved_files_selected(window)
        self.assert_opening_keeps_the_saved_files(window)

    def test_a_redraw_lists_the_directories_as_shown(self):
        directories = {}
        for _, _, key, _ in self.FIELDS:
            folder = self.root / "saved" / key.lower()
            directories[key] = (str(folder), folder)
        window = self.open_tools(directories)
        # A directory changed without saving, and a file picked from it.
        unsaved = self.root / "unsaved_sequences"
        unsaved.mkdir()
        for name in ("a_listed_first.fasta", "picked.fasta"):
            (unsaved / name).touch()
        window.dir_inputs["FASTA_DIR"].setText(str(unsaved))
        _, combo = self.dropdown(window, "Sanitize_Sequences.py", "INPUT_FASTA")
        combo.populate()
        combo.setCurrentIndex(combo.findText("picked.fasta"))
        carried = window.language_carry_over()

        replacement = self.open_tools(carried=carried)

        _, combo = self.dropdown(replacement, "Sanitize_Sequences.py", "INPUT_FASTA")
        self.assertEqual(combo.currentText(), "picked.fasta")
        self.assertEqual(replacement.language_carry_over()["document"], carried["document"])


if __name__ == "__main__":
    unittest.main()
