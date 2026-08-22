import os
import json
import pathlib
import runpy
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class CacheDropdownRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Utils  # noqa: F401 - load torch before PySide6
        from SSN_Tools import DynamicComboBox
        from PySide6.QtWidgets import QApplication
        test_app = QApplication.instance() or QApplication([])

        with mock.patch.object(QApplication, "exec", return_value=0), mock.patch.object(
            sys, "exit", return_value=None
        ):
            namespace = runpy.run_path(str(SRC / "SSN_Config.py"), run_name="__main__")

        cls.app = namespace["app"]
        cls.window = namespace["window"]
        cls.namespace = namespace
        cls.dynamic_combo_class = DynamicComboBox
        cls.window._cache_hash_request_id += 1
        for worker in cls.window._cache_hash_workers.values():
            worker.requestInterruption()
            worker.wait()
        cls.window._cache_hash_workers.clear()

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls.app.processEvents()

    def test_default_directory_layout_uses_input_and_analysis_roots(self):
        defaults = self.namespace["DIRECTORY_PROFILE_DEFAULTS"]
        self.assertEqual(
            defaults["HEADER_LIST_DIR"],
            os.path.join("Input_Files", "Header_Lists"),
        )
        self.assertEqual(
            defaults["METADATA_DIR"],
            os.path.join("Input_Files", "Meta_Data"),
        )
        self.assertEqual(
            defaults["SEQUENCE_EXPORT_DIR"],
            os.path.join("Analysis_Results", "Sequence_Export"),
        )
        self.assertEqual(
            defaults["PRINT_SAVE_DIR"],
            os.path.join("Analysis_Results", "Saved_Images"),
        )
        self.assertEqual(
            defaults["STRUCTURES_DIR"],
            os.path.join("Cache_Files", "Predicted_Structures"),
        )

    def test_requested_directory_labels_use_concise_names(self):
        expected_labels = {
            "FASTA_DIR": "Input FASTA Directory:",
            "SAVED_LAYOUT_DIR": "Layout Directory:",
            "SEQUENCE_EXPORT_DIR": "Sequence Export Directory:",
            "PRINT_SAVE_DIR": "Print Directory:",
        }
        for key, expected in expected_labels.items():
            with self.subTest(key=key):
                self.assertEqual(self.window.labels[key].text(), expected)

    def test_slider_spinboxes_match_tools_minimum_height(self):
        slider_keys = (
            "NODE_SIZE",
            "EDGE_WIDTH",
            "NODE_BOUNDARY_WIDTH",
            "EDGE_ALPHA",
            "TEXT_SIZE",
            "SPRING_K",
            "COULOMB_K",
            "COULOMB_CUTOFF",
            "DAMPING",
        )

        for key in slider_keys:
            with self.subTest(key=key):
                self.assertEqual(self.window.inputs[key].minimumHeight(), 28)

    def test_monte_carlo_rows_have_extra_top_clearance(self):
        monte_carlo_grid = self.window.findChild(
            self.namespace["QGridLayout"], "monteCarloGrid"
        )

        self.assertIsNotNone(monte_carlo_grid)
        self.assertEqual(monte_carlo_grid.contentsMargins().top(), 8)

    def test_low_resource_mode_row_has_extra_top_clearance(self):
        low_resource_row = self.window.findChild(
            self.namespace["QWidget"], "lowResourceModeRow"
        )

        self.assertIsNotNone(low_resource_row)
        self.assertEqual(low_resource_row.layout().contentsMargins().top(), 8)

    def test_spinboxes_are_not_clipped_by_parent_wrappers(self):
        from PySide6.QtWidgets import QAbstractSpinBox

        original_index = self.window.tabs.currentIndex()
        try:
            self.window.resize(1400, 900)
            self.window.show()
            for tab_index in range(self.window.tabs.count()):
                self.window.tabs.setCurrentIndex(tab_index)
                self.app.processEvents()

                for spinbox in self.window.tabs.currentWidget().findChildren(
                    QAbstractSpinBox
                ):
                    if not spinbox.isVisibleTo(self.window):
                        continue
                    with self.subTest(
                        tab=self.window.tabs.tabText(tab_index),
                        spinbox=spinbox.objectName() or type(spinbox).__name__,
                    ):
                        self.assertLessEqual(
                            spinbox.geometry().bottom(),
                            spinbox.parentWidget().rect().bottom(),
                        )
        finally:
            self.window.tabs.setCurrentIndex(original_index)
            self.app.processEvents()

    def test_opening_dropdown_finds_new_files_and_preserves_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_layout_dir = pathlib.Path(temp_dir)
            cache_folder = saved_layout_dir / "compatible-layout"
            cache_folder.mkdir()
            older = cache_folder / "older.h5"
            newer = cache_folder / "newer.h5"
            older.touch()
            newer.touch()
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            directory_input = self.window.inputs["SAVED_LAYOUT_DIR"]
            directory_input.blockSignals(True)
            directory_input.setText(str(saved_layout_dir))
            directory_input.blockSignals(False)
            self.window.current_cache_folder = str(cache_folder)

            combo = self.window.cb_cache_file
            combo.clear()
            combo.addItem(
                "older.h5", os.path.join("compatible-layout", "older.h5")
            )
            combo.addItem("(New Layout Cache)", None)
            combo.setCurrentIndex(0)

            combo.showPopup()
            combo.hidePopup()

            self.assertEqual(
                [combo.itemText(index) for index in range(combo.count())],
                ["newer.h5", "older.h5", "(New Layout Cache)"],
            )
            self.assertEqual(combo.currentText(), "older.h5")

    def test_folder_dropdown_refresh_does_not_reset_dependent_score_mode(self):
        from PySide6.QtWidgets import QComboBox

        with tempfile.TemporaryDirectory() as temp_dir:
            for filename in ("first.h5", "selected.h5"):
                pathlib.Path(temp_dir, filename).touch()

            network_combo = self.dynamic_combo_class(
                temp_dir, ".h5", include_ext=True
            )
            network_combo.populate()
            network_combo.setCurrentText("selected.h5")

            score_combo = QComboBox()
            score_combo.addItems(["global", "local"])
            score_combo.setCurrentText("local")
            observed_texts = []

            def update_score_mode(network_text):
                observed_texts.append(network_text)
                if not network_text:
                    score_combo.setCurrentIndex(-1)
                elif score_combo.currentIndex() == -1:
                    score_combo.setCurrentText("global")

            network_combo.currentTextChanged.connect(update_score_mode)

            network_combo.showPopup()
            network_combo.hidePopup()

            self.assertEqual(network_combo.currentText(), "selected.h5")
            self.assertEqual(observed_texts, [])
            self.assertEqual(score_combo.currentText(), "local")

    def test_folder_dropdown_emits_final_change_if_selection_disappears(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            remaining = pathlib.Path(temp_dir, "remaining.h5")
            selected = pathlib.Path(temp_dir, "selected.h5")
            remaining.touch()
            selected.touch()

            network_combo = self.dynamic_combo_class(
                temp_dir, ".h5", include_ext=True
            )
            network_combo.populate()
            network_combo.setCurrentText("selected.h5")
            observed_texts = []
            network_combo.currentTextChanged.connect(observed_texts.append)

            selected.unlink()
            network_combo.populate()

            self.assertEqual(network_combo.currentText(), "remaining.h5")
            self.assertEqual(observed_texts, ["remaining.h5"])

    def test_visual_color_swatch_tracks_typed_name_and_hex_code(self):
        color_input = self.window.inputs["HOVER_COLOR"]
        color_swatch = self.window.color_swatches["HOVER_COLOR"]

        color_input.setText("magenta")
        self.app.processEvents()
        self.assertIn("background-color: #ff00ff", color_swatch.styleSheet())

        color_input.setText("#123456")
        self.app.processEvents()
        self.assertIn("background-color: #123456", color_swatch.styleSheet())

    def test_profile_discovery_and_reserved_name_validation(self):
        discover = self.namespace["_discover_profile_names"]
        validate = self.namespace["_validate_profile_name"]

        with tempfile.TemporaryDirectory() as temp_dir:
            folder = pathlib.Path(temp_dir, "visual_effects")
            folder.mkdir()
            for filename in (
                "Zulu.json", "alpha.JSON", "custom.json", "default.json",
                "new.json", "notes.txt",
            ):
                (folder / filename).touch()

            self.assertEqual(
                discover(pathlib.Path(temp_dir), "visual_effects"),
                ["alpha", "Zulu"],
            )
            self.assertEqual(validate(" New Theme.json "), "New Theme")
            with self.assertRaises(ValueError):
                validate("custom")
            with self.assertRaises(ValueError):
                validate("bad/name")
            with self.assertRaises(ValueError):
                validate("ALPHA", ["alpha"])

    def test_profile_controls_default_and_new_behavior(self):
        visual_selector = self.window.profile_selectors["visual_effects"]
        input_selector = self.window.profile_selectors["inputs_outputs"]
        visual_content = self.window.profile_content_widgets["visual_effects"]
        name_input = self.window.profile_name_inputs["visual_effects"]

        self.assertEqual(
            [
                input_selector.itemText(index)
                for index in range(max(0, input_selector.count() - 2), input_selector.count())
            ],
            ["(custom)", "(new)"],
        )
        self.assertEqual(
            [
                visual_selector.itemText(index)
                for index in range(max(0, visual_selector.count() - 3), visual_selector.count())
            ],
            ["(custom)", "(default)", "(new)"],
        )

        visual_selector.setCurrentText("(default)")
        self.app.processEvents()
        self.assertFalse(visual_content.isEnabled())
        default_node_size = self.window.inputs["NODE_SIZE"].value()
        palette_type = self.namespace["QPalette"]
        disabled_group = palette_type.ColorGroup.Disabled

        node_size_label = self.window.labels["NODE_SIZE"]
        node_size_spinbox = self.window.inputs["NODE_SIZE"]
        node_size_slider = node_size_spinbox.parentWidget().findChild(
            self.namespace["QSlider"]
        )
        self.assertFalse(node_size_label.isEnabled())
        self.assertIn(
            self.namespace["PROFILE_DISABLED_LABEL_STYLESHEET"],
            node_size_label.styleSheet(),
        )
        self.assertFalse(node_size_spinbox.isEnabled())
        self.assertIn(
            self.namespace["PROFILE_DISABLED_SPINBOX_STYLESHEET"],
            node_size_spinbox.styleSheet(),
        )
        self.assertNotIn("border: 1px solid #c8c8c8", node_size_spinbox.styleSheet())
        self.assertIsNotNone(node_size_slider)
        self.assertFalse(node_size_slider.isEnabled())
        self.assertEqual(node_size_slider.styleSheet(), "")

        color_input = self.window.inputs["HOVER_COLOR"]
        pick_button = color_input.parentWidget().findChild(
            self.namespace["QPushButton"]
        )
        self.assertFalse(color_input.isEnabled())
        self.assertEqual(
            color_input.palette().color(
                disabled_group,
                palette_type.ColorRole.Base,
            ).name(),
            "#f0f0f0",
        )
        self.assertIsNotNone(pick_button)
        self.assertFalse(pick_button.isEnabled())
        self.assertEqual(
            pick_button.palette().color(
                disabled_group,
                palette_type.ColorRole.ButtonText,
            ).name(),
            "#888888",
        )

        low_resource_toggle = self.window.inputs["LOW_RESOURCE_MODE"]
        self.assertFalse(low_resource_toggle.isEnabled())
        self.assertIn(
            self.namespace["PROFILE_DISABLED_TOGGLE_STYLESHEET"],
            low_resource_toggle.styleSheet(),
        )
        default_swatch = self.window.color_swatches["HOVER_COLOR"]
        default_swatch_image = default_swatch.grab().toImage()
        self.assertEqual(
            default_swatch_image.pixelColor(
                default_swatch_image.width() // 2,
                default_swatch_image.height() // 2,
            ).name(),
            "#ffaa00",
        )

        visual_selector.setCurrentText("(new)")
        self.app.processEvents()
        self.assertTrue(visual_content.isEnabled())
        self.assertFalse(name_input.isHidden())
        self.assertEqual(self.window.inputs["NODE_SIZE"].value(), default_node_size)
        profile_row_layout = name_input.parentWidget().layout()
        self.assertEqual(profile_row_layout.stretch(0), profile_row_layout.stretch(1))

        directory_selector = self.window.profile_selectors["directories"]
        directory_selector.setCurrentText("(default)")
        self.app.processEvents()
        self.assertTrue(
            all(
                not widget.isEnabled()
                for widget in self.window.profile_content_widgets["directories"]
            )
        )
        directory_input = self.window.inputs["FASTA_DIR"]
        directory_button = directory_input.parentWidget().findChild(
            self.namespace["QPushButton"]
        )
        for widget in (
            self.window.labels["FASTA_DIR"],
            directory_input,
            directory_button,
        ):
            self.assertIsNotNone(widget)
            self.assertFalse(widget.isEnabled())
        self.assertIn(
            self.namespace["PROFILE_DISABLED_LABEL_STYLESHEET"],
            self.window.labels["FASTA_DIR"].styleSheet(),
        )
        self.assertEqual(
            directory_input.palette().color(
                disabled_group,
                palette_type.ColorRole.Text,
            ).name(),
            "#888888",
        )
        self.assertEqual(
            directory_button.palette().color(
                disabled_group,
                palette_type.ColorRole.Button,
            ).name(),
            "#f0f0f0",
        )
        self.assertTrue(self.window.inputs["SAVED_CONFIG_DIR"].isEnabled())

        physics_selector = self.window.profile_selectors["simulation_physics"]
        physics_selector.setCurrentText("(default)")
        self.app.processEvents()
        for key in (
            "PHYSICS_ENGINE",
            "DT",
            "ENABLE_PROGRESSIVE_SIMULATION",
        ):
            widget = self.window.inputs[key]
            self.assertFalse(widget.isEnabled())
            self.assertFalse(self.window.labels[key].isEnabled())
            self.assertIn(
                self.namespace["PROFILE_DISABLED_LABEL_STYLESHEET"],
                self.window.labels[key].styleSheet(),
            )
        self.assertEqual(
            self.window.inputs["PHYSICS_ENGINE"].palette().color(
                disabled_group,
                palette_type.ColorRole.Text,
            ).name(),
            "#888888",
        )
        self.assertEqual(
            self.window.inputs["DT"].palette().color(
                disabled_group,
                palette_type.ColorRole.Base,
            ).name(),
            "#f0f0f0",
        )
        self.assertIn(
            self.namespace["PROFILE_DISABLED_TOGGLE_STYLESHEET"],
            self.window.inputs["ENABLE_PROGRESSIVE_SIMULATION"].styleSheet(),
        )

        reset_buttons = [
            button.text()
            for button in self.window.findChildren(self.namespace["QPushButton"])
        ]
        self.assertNotIn("Reset to Default", reset_buttons)

        visual_selector.setCurrentText("(custom)")
        directory_selector.setCurrentText("(custom)")
        physics_selector.setCurrentText("(custom)")
        self.app.processEvents()

    def test_profile_folder_button_opens_the_specific_tab_folder(self):
        original_root = self.window.inputs["SAVED_CONFIG_DIR"].text()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                self.window.inputs["SAVED_CONFIG_DIR"].setText(temp_dir)
                with mock.patch.object(
                    self.namespace["QDesktopServices"], "openUrl", return_value=True
                ) as open_url:
                    self.window.profile_folder_buttons["inputs_outputs"].click()

                expected = pathlib.Path(temp_dir, "inputs_outputs").resolve()
                self.assertTrue(expected.is_dir())
                self.assertEqual(
                    pathlib.Path(open_url.call_args.args[0].toLocalFile()).resolve(),
                    expected,
                )
            finally:
                self.window.inputs["SAVED_CONFIG_DIR"].setText(original_root)

    def test_all_tabs_share_padding_and_separator_spacing(self):
        from PySide6.QtCore import QPoint

        original_index = self.window.tabs.currentIndex()
        checks = (
            ("inputs_outputs", None, "NODE_FASTA_FILE"),
            ("visual_effects", None, "NODE_SIZE"),
            ("simulation_physics", None, "PHYSICS_ENGINE"),
            ("directories", "SAVED_CONFIG_DIR", "FASTA_DIR"),
        )
        expected_margin = self.namespace["CONFIG_TAB_CONTENT_MARGIN"]
        expected_spacing = self.namespace["CONFIG_SEPARATOR_PADDING"]
        expected_thickness = self.namespace["CONFIG_SEPARATOR_THICKNESS"]
        expected_field_x = (
            expected_margin
            + self.namespace["CONFIG_FIELD_LABEL_WIDTH"]
            + self.namespace["CONFIG_FIELD_HORIZONTAL_SPACING"]
        )
        profile_field_positions = []
        setting_field_positions = []
        self.assertEqual(expected_spacing, 24)
        self.assertEqual(expected_thickness, 2)
        self.assertEqual(
            self.namespace["DEFAULT_SAVED_CONFIG_DIR"],
            os.path.join("Cache_Files", "Saved_Config"),
        )
        self.assertEqual(
            self.namespace["_migrate_saved_config_dir"]("Saved_Config"),
            os.path.join("Cache_Files", "Saved_Config"),
        )
        self.assertEqual(
            self.namespace["_migrate_saved_config_dir"]("D:/SSN/Profiles"),
            "D:/SSN/Profiles",
        )

        try:
            for index, (tab_id, preceding_key, first_key) in enumerate(checks):
                self.window.tabs.setCurrentIndex(index)
                self.app.processEvents()
                tab = self.window.tabs.widget(index)
                profile_label = self.window.profile_labels[tab_id]
                preceding_widget = (
                    self.window.labels[preceding_key]
                    if preceding_key is not None
                    else profile_label
                )
                separator = self.window.profile_separators[tab_id]
                setting_label = self.window.labels[first_key]
                setting_input = self.window.inputs[first_key]
                setting_field = (
                    setting_input
                    if tab_id == "simulation_physics"
                    else setting_input.parentWidget()
                )
                profile_position = profile_label.mapTo(tab, QPoint(0, 0))
                profile_field_position = self.window.profile_selectors[
                    tab_id
                ].mapTo(tab, QPoint(0, 0))
                preceding_position = preceding_widget.mapTo(tab, QPoint(0, 0))
                separator_position = separator.mapTo(tab, QPoint(0, 0))
                setting_position = setting_label.mapTo(tab, QPoint(0, 0))
                setting_field_position = setting_field.mapTo(tab, QPoint(0, 0))
                margins = tab.layout().contentsMargins()

                self.assertEqual(
                    (margins.left(), margins.top()),
                    (expected_margin, expected_margin),
                )
                self.assertEqual(profile_position.x(), expected_margin)
                self.assertEqual(profile_position.y(), expected_margin)
                self.assertEqual(profile_field_position.x(), expected_field_x)
                self.assertEqual(setting_field_position.x(), expected_field_x)
                profile_field_positions.append(profile_field_position.x())
                setting_field_positions.append(setting_field_position.x())
                self.assertEqual(separator_position.x(), expected_margin)
                self.assertEqual(separator.height(), expected_thickness)
                self.assertEqual(
                    separator.width(), tab.width() - (2 * expected_margin)
                )
                self.assertEqual(
                    separator_position.y()
                    - preceding_position.y()
                    - preceding_widget.height(),
                    expected_spacing,
                )
                self.assertEqual(
                    setting_position.y()
                    - separator_position.y()
                    - separator.height(),
                    expected_spacing,
                )
            self.assertEqual(len(set(profile_field_positions)), 1)
            self.assertEqual(len(set(setting_field_positions)), 1)
        finally:
            self.window.tabs.setCurrentIndex(original_index)
            self.app.processEvents()

    def test_cache_separator_and_conditional_name_field(self):
        from PySide6.QtCore import QPoint

        original_index = self.window.tabs.currentIndex()
        combo = self.window.cb_cache_file
        try:
            self.window.tabs.setCurrentIndex(0)
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("existing.h5", "layout/existing.h5")
            combo.addItem("(New Layout Cache)", None)
            combo.setCurrentIndex(0)
            combo.setEnabled(True)
            combo.blockSignals(False)
            self.window._toggle_new_cache_input(combo.currentText())
            self.app.processEvents()

            tab = self.window.tabs.widget(0)
            button_container = self.window.btn_stats.parentWidget()
            tracker_container = self.window.lbl_cache_tracker.parentWidget()
            separator = self.window.cache_file_separator
            cache_label = self.window.labels["TARGET_CACHE"]
            target_label = self.window.labels["TARGET_CACHE_FILE"]
            button_position = button_container.mapTo(tab, QPoint(0, 0))
            tracker_position = tracker_container.mapTo(tab, QPoint(0, 0))
            separator_position = separator.mapTo(tab, QPoint(0, 0))
            cache_position = cache_label.mapTo(tab, QPoint(0, 0))
            target_position = target_label.mapTo(tab, QPoint(0, 0))
            expected_margin = self.namespace["CONFIG_TAB_CONTENT_MARGIN"]
            expected_spacing = self.namespace["CONFIG_SEPARATOR_PADDING"]

            self.assertEqual(
                separator.height(), self.namespace["CONFIG_SEPARATOR_THICKNESS"]
            )
            self.assertEqual(separator_position.x(), expected_margin)
            self.assertEqual(
                separator.width(), tab.width() - (2 * expected_margin)
            )
            self.assertEqual(
                separator_position.y()
                - button_position.y()
                - button_container.height(),
                expected_spacing,
            )
            self.assertEqual(
                cache_position.y() - separator_position.y() - separator.height(),
                expected_spacing,
            )
            self.assertEqual(cache_label.text(), "Target Cache:")
            self.assertEqual(tracker_position.y(), cache_position.y())
            self.assertGreater(target_position.y(), tracker_position.y())

            self.assertTrue(self.window.line_new_cache.isHidden())
            combo.setCurrentText("(New Layout Cache)")
            self.app.processEvents()
            self.assertFalse(self.window.line_new_cache.isHidden())
            self.assertTrue(self.window.line_new_cache.isEnabled())
            target_layout = self.window.line_new_cache.parentWidget().layout()
            self.assertEqual(target_layout.stretch(0), target_layout.stretch(1))

            self.window.line_new_cache.setText("temporary-name")
            combo.setCurrentText("existing.h5")
            self.app.processEvents()
            self.assertTrue(self.window.line_new_cache.isHidden())
            self.assertEqual(self.window.line_new_cache.text(), "")
        finally:
            self.window.tabs.setCurrentIndex(original_index)
            self.app.processEvents()

    def test_every_setting_input_and_nested_editor_has_a_shared_tip(self):
        from PySide6.QtWidgets import QLineEdit, QWidget

        self.assertEqual(
            set(self.window.inputs) - set(self.window.tip_db_keys),
            set(),
        )
        for key, widget in self.window.inputs.items():
            with self.subTest(key=key, target="input"):
                self.assertIn(widget, self.window.tip_db)
            if key in self.window.labels:
                with self.subTest(key=key, target="label"):
                    self.assertIn(self.window.labels[key], self.window.tip_db)
            for child in widget.findChildren(QWidget):
                if isinstance(child, QLineEdit):
                    with self.subTest(key=key, target="nested editor"):
                        self.assertIn(child, self.window.tip_db)

    def test_saved_config_and_target_cache_clicks_use_shared_tip_panel(self):
        from PySide6.QtCore import QEvent

        original_tip = self.window.tip_panel.text()
        cases = []
        for tab_id in self.window.profile_selectors:
            cases.extend(
                (
                    (self.window.profile_labels[tab_id], "Saved Config:"),
                    (self.window.profile_selectors[tab_id], "Saved Config:"),
                    (self.window.profile_name_inputs[tab_id], "Saved Config:"),
                    (self.window.profile_folder_buttons[tab_id], "Saved Config:"),
                )
            )
        cases.extend(
            (
                (self.window.labels["TARGET_CACHE"], "Target Cache:"),
                (self.window.lbl_cache_tracker, "Target Cache:"),
                (self.window.btn_open_cache, "Target Cache:"),
            )
        )
        try:
            for target, expected_tip in cases:
                with self.subTest(target=type(target).__name__):
                    self.assertIn(target, self.window.tip_db)
                    self.window.tip_panel.setText("sentinel")
                    self.window.eventFilter(
                        target,
                        QEvent(QEvent.Type.MouseButtonPress),
                    )
                    self.assertIn(expected_tip, self.window.tip_panel.text())
        finally:
            self.window.tip_panel.setText(original_tip)

    def test_native_tooltip_popup_is_redirected_to_shared_tip_panel(self):
        from PySide6.QtCore import QEvent, QPoint
        from PySide6.QtGui import QHelpEvent

        original_tip = self.window.tip_panel.text()
        try:
            handled = self.window.eventFilter(
                self.window.btn_open_cache,
                QHelpEvent(
                    QEvent.Type.ToolTip,
                    QPoint(1, 1),
                    QPoint(1, 1),
                ),
            )

            self.assertTrue(handled)
            self.assertIn("Target Cache:", self.window.tip_panel.text())
        finally:
            self.window.tip_panel.setText(original_tip)

    def test_save_only_reports_success_in_tooltip_without_popup(self):
        original_tip = self.window.tip_panel.text()
        try:
            with mock.patch.object(
                self.window,
                "_prepare_profile_writes",
                return_value=([], dict(self.window._custom_settings), [], []),
            ), mock.patch.object(
                self.namespace["QMessageBox"], "information"
            ) as information, mock.patch.object(
                self.namespace["QMessageBox"], "critical"
            ) as critical:
                self.window.save_only()

            self.assertIn(
                "Settings saved successfully.", self.window.tip_panel.text()
            )
            information.assert_not_called()
            critical.assert_not_called()
        finally:
            self.window.tip_panel.setText(original_tip)

    def test_save_reports_default_tabs_in_tooltip(self):
        original_tip = self.window.tip_panel.text()
        try:
            with mock.patch.object(
                self.window,
                "_prepare_profile_writes",
                return_value=(
                    [],
                    dict(self.window._custom_settings),
                    [],
                    ["visual_effects", "directories"],
                ),
            ), mock.patch.object(
                self.namespace["QMessageBox"], "critical"
            ) as critical:
                self.assertTrue(self.window.save_settings())

            self.assertIn(
                "Settings saved successfully. Built-in default settings were "
                "left unchanged for: Visual Effects, Directories.",
                self.window.tip_panel.text(),
            )
            critical.assert_not_called()
        finally:
            self.window.tip_panel.setText(original_tip)

    def test_new_profile_without_name_reports_in_tooltip_without_popup(self):
        original_tip = self.window.tip_panel.text()
        selector = self.window.profile_selectors["visual_effects"]
        name_input = self.window.profile_name_inputs["visual_effects"]
        original_selection = selector.currentText()
        original_name = name_input.text()
        try:
            selector.setCurrentText("(new)")
            name_input.clear()
            with mock.patch.object(
                self.namespace["QMessageBox"], "critical"
            ) as critical:
                self.assertFalse(self.window.save_settings())

            self.assertIn(
                "Failed to save settings: Enter a profile name.",
                self.window.tip_panel.text(),
            )
            critical.assert_not_called()
        finally:
            selector.setCurrentText(original_selection)
            name_input.setText(original_name)
            self.window.tip_panel.setText(original_tip)

    def test_malformed_named_profile_keeps_current_visual_state(self):
        selector = self.window.profile_selectors["visual_effects"]
        original_root = self.window.inputs["SAVED_CONFIG_DIR"].text()
        original_custom = dict(self.window._custom_settings)

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                profile_folder = pathlib.Path(temp_dir, "visual_effects")
                profile_folder.mkdir()
                (profile_folder / "broken.json").write_text(
                    '{"NODE_SIZE": 999}', encoding="utf-8"
                )
                self.window.inputs["SAVED_CONFIG_DIR"].setText(temp_dir)
                self.window._saved_config_directory_committed()
                self.window.inputs["NODE_SIZE"].setValue(12)
                self.window._refresh_profile_combo("visual_effects")

                with mock.patch.object(
                    self.namespace["QMessageBox"], "critical"
                ) as critical:
                    selector.setCurrentText("broken")
                    self.app.processEvents()

                self.assertTrue(critical.called)
                self.assertEqual(selector.currentText(), "(custom)")
                self.assertEqual(self.window.inputs["NODE_SIZE"].value(), 12)
            finally:
                self.window._custom_settings = original_custom
                self.window.inputs["SAVED_CONFIG_DIR"].setText(original_root)
                self.window._saved_config_directory_committed()

    def test_save_routes_custom_named_and_default_tabs(self):
        globals_dict = self.window.save_settings.__globals__
        original_settings_file = globals_dict["DEFAULT_SETTINGS_FILE"]
        original_root = self.window.inputs["SAVED_CONFIG_DIR"].text()
        original_custom = dict(self.window._custom_settings)

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                root = pathlib.Path(temp_dir, "profiles")
                visual_folder = root / "visual_effects"
                visual_folder.mkdir(parents=True)
                visual_profile = dict(self.namespace["VISUAL_PROFILE_DEFAULTS"])
                visual_profile["TEXT_SIZE"] = 12
                (visual_folder / "theme.json").write_text(
                    json.dumps(visual_profile), encoding="utf-8"
                )

                settings_file = pathlib.Path(temp_dir, "viewer_settings.json")
                globals_dict["DEFAULT_SETTINGS_FILE"] = str(settings_file)
                self.window._custom_settings = {
                    "TEXT_SIZE": "11",
                    "SPRING_K": "7.0",
                    "SAVED_CONFIG_DIR": str(root),
                }
                self.window.inputs["SAVED_CONFIG_DIR"].setText(str(root))
                self.window._saved_config_directory_committed()

                self.window.profile_selectors["visual_effects"].setCurrentText("theme")
                self.window.inputs["TEXT_SIZE"].setValue(13)
                self.window.profile_selectors["simulation_physics"].setCurrentText(
                    "(default)"
                )

                with mock.patch.object(self.namespace["QMessageBox"], "critical"):
                    self.assertTrue(self.window.save_settings())

                saved_custom = json.loads(
                    settings_file.read_text(encoding="utf-8")
                )
                saved_named = json.loads(
                    (visual_folder / "theme.json").read_text(encoding="utf-8")
                )
                self.assertEqual(saved_custom["TEXT_SIZE"], "11")
                self.assertEqual(saved_custom["SPRING_K"], "7.0")
                self.assertEqual(saved_custom["SAVED_CONFIG_DIR"], str(root))
                self.assertEqual(saved_named["TEXT_SIZE"], "13")
                self.assertNotIn("SAVED_CONFIG_DIR", saved_named)
                self.assertFalse((root / "simulation_physics" / "default.json").exists())
                self.assertIn(
                    "Built-in default settings were left unchanged for: "
                    "Simulation &amp; Physics.",
                    self.window.tip_panel.text(),
                )
            finally:
                globals_dict["DEFAULT_SETTINGS_FILE"] = original_settings_file
                self.window._custom_settings = original_custom
                self.window.inputs["SAVED_CONFIG_DIR"].setText(original_root)
                self.window._saved_config_directory_committed()

    def test_new_profile_is_created_selected_and_cannot_be_duplicated(self):
        globals_dict = self.window.save_settings.__globals__
        original_settings_file = globals_dict["DEFAULT_SETTINGS_FILE"]
        original_root = self.window.inputs["SAVED_CONFIG_DIR"].text()
        original_custom = dict(self.window._custom_settings)

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                root = pathlib.Path(temp_dir, "profiles")
                globals_dict["DEFAULT_SETTINGS_FILE"] = str(
                    pathlib.Path(temp_dir, "viewer_settings.json")
                )
                self.window._custom_settings = {"SAVED_CONFIG_DIR": str(root)}
                self.window.inputs["SAVED_CONFIG_DIR"].setText(str(root))
                self.window._saved_config_directory_committed()

                selector = self.window.profile_selectors["visual_effects"]
                name_input = self.window.profile_name_inputs["visual_effects"]
                selector.setCurrentText("(new)")
                name_input.setText("publication")
                self.window.inputs["NODE_SIZE"].setValue(14)
                self.assertTrue(self.window.save_settings())

                created = root / "visual_effects" / "publication.json"
                self.assertTrue(created.exists())
                self.assertEqual(selector.currentText(), "publication")
                self.assertTrue(name_input.isHidden())
                self.assertIn("Created profile(s): Visual Effects:", self.window.tip_panel.text())
                self.assertIn("publication", self.window.tip_panel.text())

                selector.setCurrentText("(new)")
                name_input.setText("publication")
                with mock.patch.object(
                    self.namespace["QMessageBox"], "critical"
                ) as critical:
                    self.assertFalse(self.window.save_settings())
                critical.assert_not_called()
                self.assertIn("Failed to save settings:", self.window.tip_panel.text())
                self.assertIn("publication", self.window.tip_panel.text())
                self.assertIn("already exists.", self.window.tip_panel.text())
                self.assertEqual(selector.currentText(), "(new)")
            finally:
                globals_dict["DEFAULT_SETTINGS_FILE"] = original_settings_file
                self.window._custom_settings = original_custom
                self.window.inputs["SAVED_CONFIG_DIR"].setText(original_root)
                self.window._saved_config_directory_committed()

    def test_layout_export_contains_only_generation_settings_and_exact_name(self):
        generation_values = {
            "NODE_FASTA_FILE": "Input_Files/Sequence_Sets/example.fasta",
            "INPUT_HDF5": "Input_Files/Networks_EValues/example.h5",
            "ALIGNMENT_SCORE": "global",
            "NORM_MODE": "alignment_length",
            "UMAP_MODE": False,
            "UMAP_NEIGHBORS": 15,
            "UMAP_MIN_DIST": 0.1,
            "PHYSICS_ENGINE": "Molecular Dynamics (Style)",
            "LAYOUT_DEVICE_SELECTION": "auto",
            "SPRING_K": 5.0,
            "COULOMB_K": 10.0,
            "COULOMB_CUTOFF": 30.0,
            "DAMPING": 0.9,
            "DT": 0.005,
            "MAX_STEPS": 10000,
            "RMSD_THRESHOLD": 0.005,
            "PERCENTAGE_DROP_THRESHOLD": 0.1,
            "RMSD_WINDOW": 50,
            "ENABLE_PROGRESSIVE_SIMULATION": False,
            "PACKING_GEOMETRY": "Square",
            "PACKING_GRID_SIZE": 20.0,
            "SGLD_MIN_K": 20,
            "SGLD_K_PERCENT": 0.01,
            "SGLD_START_TEMP": 1.5,
            "SGLD_NOISE_SCALE": 1.0,
            "NODE_SIZE": 10,
            "MSA_FILE": "example.fasta",
            "PRINT_SAVE_DIR": "Analysis_Results/Saved_Images",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            target_json = temp_path / "exported.json"
            method_globals = self.window.export_layout_settings.__func__.__globals__
            with mock.patch.object(
                self.window, "collect_data", return_value=generation_values
            ), mock.patch.object(
                self.window, "_selected_new_cache_filename", return_value="exact.h5"
            ), mock.patch.object(
                self.window.spin_thresh, "optionalValue", return_value=0.1
            ), mock.patch.object(
                self.window.spin_top, "optionalValue", return_value=None
            ), mock.patch.object(
                self.window.cb_layout_device, "currentData", return_value="auto"
            ), mock.patch.object(
                self.window.inputs["SAVED_LAYOUT_DIR"],
                "text",
                return_value=str(temp_path / "layouts"),
            ), mock.patch.object(
                self.window, "_cache_launch_allowed", True
            ), mock.patch.object(
                self.window, "current_cache_folder", str(temp_path / "target")
            ), mock.patch.dict(
                method_globals, {"PROJECT_ROOT": temp_path}
            ):
                settings = self.window._collect_layout_generation_settings()
                document = settings.to_document(project_root=temp_path)
                payload = document["Layout_Cache_Generator.py"]
                self.assertEqual(payload["CACHE_FILENAME"], "exact.h5")
                self.assertIs(payload["UMAP_MODE"], False)
                self.assertIsInstance(payload["MAX_STEPS"], int)
                self.assertNotIn("NODE_SIZE", payload)
                self.assertNotIn("MSA_FILE", payload)
                self.assertNotIn("PRINT_SAVE_DIR", payload)
                self.assertEqual(
                    set(document["DIRECTORIES"]), {"SAVED_LAYOUT_DIR"}
                )

                with mock.patch.object(
                    self.namespace["QFileDialog"],
                    "getSaveFileName",
                    return_value=(str(target_json), "JSON Files (*.json)"),
                ), mock.patch.object(
                    self.namespace["QMessageBox"], "information"
                ) as information, mock.patch.object(
                    self.namespace["QMessageBox"], "critical"
                ) as critical:
                    self.window.export_layout_settings()

            exported = json.loads(target_json.read_text(encoding="utf-8"))
            self.assertEqual(exported, document)
            information.assert_called_once()
            self.assertIn(
                "Layout_Cache_Generator.py",
                information.call_args.args[2],
            )
            critical.assert_not_called()

    def test_layout_export_button_is_enabled_only_for_new_cache_generation(self):
        self.assertIn("#2196F3", self.window.btn_export_layout.styleSheet())
        self.assertEqual(
            self.window.btn_export_layout.sizeHint().height(),
            self.window.btn_save_run.sizeHint().height(),
        )
        self.assertEqual(
            self.window.btn_export_layout.sizeHint().height(),
            self.window.btn_check.sizeHint().height(),
        )
        with mock.patch.object(self.window, "_cache_launch_allowed", True):
            self.window._toggle_new_cache_input("(New Layout Cache)")
            self.assertTrue(self.window.btn_export_layout.isEnabled())
            self.window._toggle_new_cache_input("version_00.h5")
            self.assertFalse(self.window.btn_export_layout.isEnabled())
        self.window._toggle_new_cache_input(self.window.cb_cache_file.currentText())


if __name__ == "__main__":
    unittest.main()
