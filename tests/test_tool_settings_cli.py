import importlib.util
import json
import os
import pathlib
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
            DEFAULT_DIRECTORY_PATHS["SETTING_EXPORT_DIR"],
            os.path.join("Cache_Files", "Tool_Settings"),
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
            os.path.join(PROJECT_ROOT, "Input_Files", "tools_settings.json"),
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
        self.assertEqual(
            module.EMBED_DIR,
            os.path.normpath(PROJECT_ROOT / "portable_embeddings"),
        )


class ToolExportGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        from SSN_Tools import ToolsGUI

        cls.app = QApplication.instance() or QApplication([])
        cls.tools_gui_class = ToolsGUI

    def test_export_filename_validation(self):
        normalize = self.tools_gui_class._normalized_export_filename
        self.assertEqual(normalize("analysis"), "analysis.json")
        self.assertEqual(normalize("analysis.JSON"), "analysis.JSON")
        for invalid in ("", "../escape", "bad:name", "CON", "CON.txt", "trailing."):
            with self.subTest(name=invalid), self.assertRaises(ValueError):
                normalize(invalid)

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
                "FASTA_DIR": QLineEdit("current_sequences"),
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
            self.assertEqual(payload["DIRECTORIES"], {"FASTA_DIR": "current_sequences"})
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
