# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Configuration window inputs, directories, profiles, separators, and save reporting."""

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from tests.config_cache_fixtures import ROOT, SharedConfigWindowCase, left_press


class ConfigInputsTests(SharedConfigWindowCase):
    def test_default_directory_layout_uses_input_and_analysis_roots(self):
        defaults = self.namespace["DIRECTORY_PROFILE_DEFAULTS"]
        self.assertEqual(defaults["INPUT_FILE_DIR"], "Input_Files")
        self.assertEqual(defaults["CACHE_FILE_DIR"], "Cache_Files")
        self.assertEqual(defaults["ANALYSIS_RESULT_DIR"], "Analysis_Results")
        # The embeddings folder is the fourth reusable root: beside Input_Files,
        # not in it, and what $embedding$ stands for.
        self.assertEqual(defaults["EMBED_DIR"], "Embeddings")
        self.assertEqual(
            list(defaults)[:4],
            ["INPUT_FILE_DIR", "CACHE_FILE_DIR", "ANALYSIS_RESULT_DIR", "EMBED_DIR"],
        )
        self.assertEqual(
            self.namespace["DIRECTORY_PATH_ALIASES"],
            {
                "$input_file$": "INPUT_FILE_DIR",
                "$cache_file$": "CACHE_FILE_DIR",
                "$analysis_result$": "ANALYSIS_RESULT_DIR",
                "$embedding$": "EMBED_DIR",
            },
        )
        self.assertEqual(self.namespace["EMBEDDING_ALIAS"], "$embedding$")
        self.assertEqual(self.namespace["EMBED_DIR"], defaults["EMBED_DIR"])
        self.assertEqual(
            defaults["HEADER_LIST_DIR"],
            os.path.join("$input_file$", "Header_Lists"),
        )
        self.assertEqual(
            defaults["METADATA_DIR"],
            os.path.join("$input_file$", "Meta_Data"),
        )
        self.assertEqual(
            defaults["SAVED_LAYOUT_DIR"],
            os.path.join("$cache_file$", "Saved_Layouts"),
        )
        self.assertEqual(
            defaults["SETTING_EXPORT_DIR"],
            os.path.join("$cache_file$", "Exported_Settings"),
        )
        self.assertTrue(
            self.namespace["DEPRECATED_DIRECTORY_KEYS"].isdisjoint(defaults)
        )

    def test_directory_alias_resolver_is_exact_leading_and_cross_separator(self):
        resolve = self.namespace["resolve_directory_path"]
        bases = {
            "INPUT_FILE_DIR": os.path.join("custom", "inputs"),
            "CACHE_FILE_DIR": os.path.abspath(os.path.join("custom", "cache")),
            "ANALYSIS_RESULT_DIR": os.path.join("custom", "results"),
            "EMBED_DIR": os.path.join("custom", "embeddings"),
        }
        self.assertEqual(
            resolve(r"$input_file$\Sequence_Sets", bases),
            os.path.normpath(os.path.join("custom", "inputs", "Sequence_Sets")),
        )
        self.assertEqual(
            resolve(r"$embedding$\sub", bases),
            os.path.normpath(os.path.join("custom", "embeddings", "sub")),
        )
        self.assertEqual(
            resolve("$embedding$/sub/deeper", bases),
            os.path.normpath(os.path.join("custom", "embeddings", "sub", "deeper")),
        )
        self.assertEqual(resolve("$embedding$", bases), bases["EMBED_DIR"])
        self.assertEqual(
            resolve("$analysis_result$/Sequence_Logos", bases),
            os.path.normpath(os.path.join("custom", "results", "Sequence_Logos")),
        )
        self.assertEqual(
            resolve("$cache_file$", bases),
            bases["CACHE_FILE_DIR"],
        )
        for unchanged in (
            "$unknown$/folder",
            "prefix/$input_file$/folder",
            "prefix/$embedding$/folder",
            "$input_file$suffix/folder",
            "$embedding$suffix/folder",
            os.path.abspath(os.path.join("literal", "folder")),
        ):
            with self.subTest(unchanged=unchanged):
                self.assertEqual(resolve(unchanged, bases), unchanged)

    def test_legacy_child_defaults_are_linked_to_base_tokens(self):
        migrate = self.namespace["_migrate_default_directory_path"]
        defaults = self.namespace["DIRECTORY_PROFILE_DEFAULTS"]
        legacy_defaults = self.namespace["LEGACY_DEFAULT_DIRECTORY_PATHS"]

        for key, legacy_value in legacy_defaults.items():
            for stored_value in (
                legacy_value.replace("\\", "/"),
                legacy_value.replace("/", "\\"),
            ):
                with self.subTest(key=key, stored_value=stored_value):
                    self.assertEqual(migrate(key, stored_value), defaults[key])

        custom_relative = os.path.join("My_Inputs", "Sequence_Sets")
        custom_absolute = os.path.abspath(os.path.join("My_Inputs", "Sequence_Sets"))
        self.assertEqual(migrate("FASTA_DIR", custom_relative), custom_relative)
        self.assertEqual(migrate("FASTA_DIR", custom_absolute), custom_absolute)

    def test_shared_viewer_settings_file_is_at_project_root(self):
        self.assertEqual(
            pathlib.Path(self.namespace["DEFAULT_SETTINGS_FILE"]),
            ROOT / "viewer_settings.json",
        )

    def test_requested_directory_labels_use_concise_names(self):
        expected_labels = {
            "INPUT_FILE_DIR": "Input File Directory:",
            "CACHE_FILE_DIR": "Cache File Directory:",
            "ANALYSIS_RESULT_DIR": "Analysis Results Directory:",
            "EMBED_DIR": "Embedding Directory:",
            "FASTA_DIR": "Input FASTA Directory:",
            "SAVED_LAYOUT_DIR": "Layout Directory:",
            "SETTING_EXPORT_DIR": "Setting Export Directory:",
        }
        for key, expected in expected_labels.items():
            with self.subTest(key=key):
                self.assertEqual(self.window.labels[key].text(), expected)
        for key in self.namespace["DEPRECATED_DIRECTORY_KEYS"]:
            self.assertNotIn(key, self.window.inputs)
            self.assertNotIn(key, self.window.labels)

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

    def test_physics_tab_exposes_only_ssn_physics_controls(self):
        from tests.config_gui_loader import choose_layout
        choose_layout(self.window, "ssn")
        self.window.profile_selectors["simulation_physics"].setCurrentText("(new)")
        device = self.window.inputs["LAYOUT_DEVICE_SELECTION"]
        progressive = self.window.inputs["ENABLE_PROGRESSIVE_SIMULATION"]
        device.setCurrentIndex(device.findData("auto"))
        progressive.setChecked(True)
        self.app.processEvents()

        obsolete = self.namespace["OBSOLETE_LAYOUT_ENGINE_KEYS"]
        for key in obsolete:
            self.assertNotIn(key, self.window.inputs)
            self.assertNotIn(key, self.window.labels)
        for key in (
            "DAMPING",
            "DT",
            "MAX_STEPS",
            "RMSD_THRESHOLD",
            "PERCENTAGE_DROP_THRESHOLD",
            "RMSD_WINDOW",
            "ENABLE_PROGRESSIVE_SIMULATION",
        ):
            self.assertTrue(self.window.inputs[key].isEnabled())
            self.assertTrue(self.window.labels[key].isEnabled())
        self.assertTrue(device.isEnabled())
        self.assertTrue(self.window.labels["LAYOUT_DEVICE_SELECTION"].isEnabled())
        self.assertEqual(device.currentData(), "auto")
        self.assertTrue(progressive.isChecked())
        collected = self.window.collect_data()
        self.assertEqual(collected["LAYOUT_DEVICE_SELECTION"], "auto")
        self.assertTrue(collected["ENABLE_PROGRESSIVE_SIMULATION"])

    def test_auto_step_size_button_greys_out_the_step_size_field(self):
        from tests.config_gui_loader import choose_layout
        choose_layout(self.window, "ssn")
        physics_selector = self.window.profile_selectors["simulation_physics"]
        physics_selector.setCurrentText("(new)")
        button = self.window.inputs["AUTO_DT"]
        field = self.window.inputs["DT"]
        self.addCleanup(physics_selector.setCurrentText, "(custom)")
        self.addCleanup(button.setChecked, False)

        # The button starts the Step Size cell, where the other fields start,
        # and the cell needs no more room than a plain field, so its row wraps
        # no sooner than the others.
        cell = button.parentWidget()
        self.assertIs(cell.layout().itemAt(0).widget(), button)
        self.assertIs(cell.layout().itemAt(1).widget(), field)
        self.assertEqual(
            cell.layout().spacing(),
            self.namespace["CONFIG_FIELD_HORIZONTAL_SPACING"],
        )

        def minimum_width(widget):
            return widget.minimumSizeHint().expandedTo(widget.minimumSize()).width()

        self.assertEqual(
            minimum_width(cell), minimum_width(self.window.inputs["MAX_STEPS"])
        )

        button.setChecked(False)
        self.app.processEvents()
        self.assertEqual(button.text(), "Auto OFF")
        self.assertTrue(field.isEnabled())
        self.assertIs(self.window.collect_data()["AUTO_DT"], False)

        button.setChecked(True)
        self.app.processEvents()
        self.assertEqual(button.text(), "Auto ON")
        self.assertFalse(field.isEnabled())
        self.assertTrue(self.window.labels["DT"].isEnabled())
        self.assertIs(self.window.collect_data()["AUTO_DT"], True)
        self.assertIs(
            self.window._collect_tab_profile_data("simulation_physics")["AUTO_DT"], True
        )

        # The read-only built-in profile turns Auto off and locks the button.
        physics_selector.setCurrentText("(default)")
        self.app.processEvents()
        self.assertFalse(button.isChecked())
        self.assertFalse(button.isEnabled())
        self.assertEqual(button.styleSheet(), self.switch_stylesheet)

        physics_selector.setCurrentText("(new)")
        button.setChecked(False)
        self.app.processEvents()
        self.assertTrue(field.isEnabled())

    def test_low_resource_mode_sits_under_a_padded_separator(self):
        separator = self.window.findChild(self.namespace["QFrame"], "lowResourceSeparator")
        saved_config = self.window.profile_separators["visual_effects"]

        self.assertIsNotNone(separator)
        self.assertEqual(separator.property("role"), "separator")
        self.assertEqual(separator.parentWidget().layout().contentsMargins(),
                         saved_config.parentWidget().layout().contentsMargins())
        # No title under it, as the tab's other sections have none.
        self.assertIsNone(self.window.findChild(self.namespace["QLabel"], "lowResourceTitle"))

    def test_spinboxes_are_not_clipped_by_parent_wrappers(self):
        from PySide6.QtWidgets import QAbstractSpinBox

        original_index = self.window.tabs.currentIndex()
        checked = 0
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
                    checked += 1
                    with self.subTest(
                        tab=self.window.tabs.tabText(tab_index),
                        spinbox=spinbox.objectName() or type(spinbox).__name__,
                    ):
                        self.assertLessEqual(
                            spinbox.geometry().bottom(),
                            spinbox.parentWidget().rect().bottom(),
                        )
            # The visibility filter above must let spin boxes through, or nothing is checked.
            self.assertGreater(checked, 0)
        finally:
            self.window.tabs.setCurrentIndex(original_index)
            self.app.processEvents()

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
        from PySide6.QtGui import QPalette as palette_type
        disabled_group = palette_type.ColorGroup.Disabled

        node_size_label = self.window.labels["NODE_SIZE"]
        node_size_spinbox = self.window.inputs["NODE_SIZE"]
        node_size_slider = node_size_spinbox.parentWidget().findChild(
            self.namespace["QSlider"]
        )
        # A locked profile's widgets are disabled; the theme's :disabled rules
        # grey them, so none carries a stylesheet of its own.
        self.assertFalse(node_size_label.isEnabled())
        self.assertEqual(node_size_label.styleSheet(), "")
        self.assertFalse(node_size_spinbox.isEnabled())
        self.assertEqual(node_size_spinbox.styleSheet(), "")
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
            self.tokens["field_disabled"],
        )
        self.assertIsNotNone(pick_button)
        self.assertFalse(pick_button.isEnabled())
        self.assertEqual(
            pick_button.palette().color(
                disabled_group,
                palette_type.ColorRole.ButtonText,
            ).name(),
            self.tokens["text_disabled"],
        )

        for key in ("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_ANIMATIONS", "LOW_RESOURCE_DRAWING",
                    "LOW_RESOURCE_WEBSERVER", "SHOW_FPS"):
            low_resource_toggle = self.window.inputs[key]
            self.assertFalse(low_resource_toggle.isEnabled(), key)
            self.assertEqual(low_resource_toggle.styleSheet(), self.switch_stylesheet, key)
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
        self.assertEqual(self.window.labels["FASTA_DIR"].styleSheet(), "")
        self.assertEqual(
            directory_input.palette().color(
                disabled_group,
                palette_type.ColorRole.Text,
            ).name(),
            self.tokens["text_disabled"],
        )
        self.assertEqual(
            directory_button.palette().color(
                disabled_group,
                palette_type.ColorRole.Button,
            ).name(),
            self.tokens["field_disabled"],
        )
        self.assertTrue(self.window.inputs["SAVED_CONFIG_DIR"].isEnabled())

        physics_selector = self.window.profile_selectors["simulation_physics"]
        physics_selector.setCurrentText("(default)")
        self.app.processEvents()
        for key in (
            "LAYOUT_DEVICE_SELECTION",
            "DT",
            "ENABLE_PROGRESSIVE_SIMULATION",
        ):
            widget = self.window.inputs[key]
            self.assertFalse(widget.isEnabled())
            self.assertFalse(self.window.labels[key].isEnabled())
            self.assertEqual(self.window.labels[key].styleSheet(), "")
        self.assertEqual(
            self.window.inputs["LAYOUT_DEVICE_SELECTION"].palette().color(
                disabled_group,
                palette_type.ColorRole.Text,
            ).name(),
            self.tokens["text_disabled"],
        )
        self.assertEqual(
            self.window.inputs["DT"].palette().color(
                disabled_group,
                palette_type.ColorRole.Base,
            ).name(),
            self.tokens["field_disabled"],
        )
        self.assertEqual(
            self.window.inputs["ENABLE_PROGRESSIVE_SIMULATION"].styleSheet(),
            self.switch_stylesheet,
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
                    self.namespace["Desktop_Open"], "open_path", return_value=True
                ) as open_path:
                    self.window.profile_folder_buttons["inputs_outputs"].click()

                expected = pathlib.Path(temp_dir, "inputs_outputs").resolve()
                self.assertTrue(expected.is_dir())
                self.assertEqual(
                    pathlib.Path(open_path.call_args.args[0]).resolve(),
                    expected,
                )
            finally:
                self.window.inputs["SAVED_CONFIG_DIR"].setText(original_root)

    def test_saved_config_token_follows_cache_base(self):
        original_root = self.window.inputs["SAVED_CONFIG_DIR"].text()
        original_cache_base = self.window.inputs["CACHE_FILE_DIR"].text()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                self.window.inputs["SAVED_CONFIG_DIR"].setText(
                    os.path.join("$cache_file$", "Saved_Config")
                )
                self.window.inputs["CACHE_FILE_DIR"].setText(temp_dir)
                with mock.patch.object(
                    self.namespace["Desktop_Open"], "open_path", return_value=True
                ) as open_path:
                    self.window.profile_folder_buttons["directories"].click()

                expected = pathlib.Path(temp_dir, "Saved_Config", "directories").resolve()
                self.assertTrue(expected.is_dir())
                self.assertEqual(
                    pathlib.Path(open_path.call_args.args[0]).resolve(),
                    expected,
                )
            finally:
                self.window.inputs["CACHE_FILE_DIR"].setText(original_cache_base)
                self.window.inputs["SAVED_CONFIG_DIR"].setText(original_root)

    def test_directory_open_buttons_precede_browse_and_open_selected_folder(self):
        expected_keys = {
            "SAVED_CONFIG_DIR",
            *self.namespace["DIRECTORY_PROFILE_DEFAULTS"],
        }
        self.assertEqual(set(self.window.directory_open_buttons), expected_keys)

        for key, button in self.window.directory_open_buttons.items():
            with self.subTest(key=key):
                row_layout = button.parentWidget().layout()
                widgets = [
                    row_layout.itemAt(index).widget()
                    for index in range(row_layout.count())
                ]
                button_index = widgets.index(button)
                self.assertIs(widgets[button_index - 1], self.window.inputs[key])
                self.assertEqual(widgets[button_index + 1].text(), "Browse...")

        line_edit = self.window.inputs["FASTA_DIR"]
        original_path = line_edit.text()
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_folder = pathlib.Path(temp_dir, "selected", "fasta")
            try:
                line_edit.setText(str(selected_folder))
                with mock.patch.object(
                    self.namespace["Desktop_Open"], "open_path", return_value=True
                ) as open_path:
                    self.window.directory_open_buttons["FASTA_DIR"].click()

                self.assertTrue(selected_folder.is_dir())
                self.assertEqual(
                    pathlib.Path(open_path.call_args.args[0]).resolve(),
                    selected_folder.resolve(),
                )
            finally:
                line_edit.setText(original_path)

    def test_directory_rows_start_with_bases_and_aliases_rebase_live(self):
        ordered_keys = (
            "INPUT_FILE_DIR",
            "CACHE_FILE_DIR",
            "ANALYSIS_RESULT_DIR",
            "EMBED_DIR",
            "FASTA_DIR",
            "MSA_DIR",
            "HDF5_DIR",
            "METADATA_DIR",
            "HEADER_LIST_DIR",
            "SAVED_LAYOUT_DIR",
            "SETTING_EXPORT_DIR",
        )
        form_layout = self.window.labels[ordered_keys[0]].parentWidget().layout()
        rows = [form_layout.getWidgetPosition(self.window.labels[key])[0] for key in ordered_keys]
        self.assertEqual(rows, sorted(rows))

        original_base = self.window.inputs["INPUT_FILE_DIR"].text()
        original_fasta = self.window.inputs["FASTA_DIR"].text()
        with tempfile.TemporaryDirectory() as temp_dir:
            sequence_dir = pathlib.Path(temp_dir, "Sequence_Sets")
            sequence_dir.mkdir()
            (sequence_dir / "rebased.fasta").write_text(
                ">rebased\nAAAA\n", encoding="utf-8"
            )
            try:
                self.window.inputs["FASTA_DIR"].setText(
                    os.path.join("$input_file$", "Sequence_Sets")
                )
                self.window.inputs["INPUT_FILE_DIR"].setText(temp_dir)
                self.app.processEvents()

                self.assertGreaterEqual(
                    self.window.cb_fasta.findText("rebased.fasta"), 0
                )
                with mock.patch.object(
                    self.namespace["Desktop_Open"], "open_path", return_value=True
                ) as open_path:
                    self.window.directory_open_buttons["FASTA_DIR"].click()
                self.assertEqual(
                    pathlib.Path(open_path.call_args.args[0]).resolve(),
                    sequence_dir.resolve(),
                )
            finally:
                self.window.inputs["INPUT_FILE_DIR"].setText(original_base)
                self.window.inputs["FASTA_DIR"].setText(original_fasta)
                self.app.processEvents()

    def test_directory_base_separator_matches_shared_geometry(self):
        from PySide6.QtCore import QPoint

        original_index = self.window.tabs.currentIndex()
        directory_index = next(
            index
            for index in range(self.window.tabs.count())
            if self.window.tabs.tabText(index) == "Directories"
        )
        try:
            self.window.tabs.setCurrentIndex(directory_index)
            self.app.processEvents()
            tab = self.window.tabs.widget(directory_index).widget()
            separator = self.window.directory_base_separator
            base_field = self.window.inputs["EMBED_DIR"].parentWidget()
            first_child = self.window.labels["FASTA_DIR"]
            separator_position = separator.mapTo(tab, QPoint(0, 0))
            base_position = base_field.mapTo(tab, QPoint(0, 0))
            child_position = first_child.mapTo(tab, QPoint(0, 0))
            expected_spacing = self.namespace["CONFIG_SEPARATOR_PADDING"]

            self.assertEqual(separator.height(), self.namespace["CONFIG_SEPARATOR_THICKNESS"])
            self.assertEqual(
                separator.width(),
                tab.width() - (2 * self.namespace["CONFIG_TAB_CONTENT_MARGIN"]),
            )
            self.assertEqual(
                separator_position.y() - base_position.y() - base_field.height(),
                expected_spacing,
            )
            self.assertEqual(
                child_position.y() - separator_position.y() - separator.height(),
                expected_spacing,
            )
        finally:
            self.window.tabs.setCurrentIndex(original_index)
            self.app.processEvents()

    def test_embedding_directory_is_the_fourth_root_above_the_separator(self):
        window = self.window
        roots = ("INPUT_FILE_DIR", "CACHE_FILE_DIR", "ANALYSIS_RESULT_DIR", "EMBED_DIR")
        form_layout = window.labels["INPUT_FILE_DIR"].parentWidget().layout()

        def row_of(widget):
            return form_layout.getWidgetPosition(widget)[0]

        root_rows = [row_of(window.labels[key]) for key in roots]
        separator_row = row_of(window.directory_base_separator.parentWidget())
        # The four roots are consecutive rows and the separator closes them.
        self.assertEqual(root_rows, list(range(root_rows[0], root_rows[0] + 4)))
        self.assertEqual(separator_row, root_rows[-1] + 1)
        for key in (
            "FASTA_DIR", "MSA_DIR", "HDF5_DIR", "METADATA_DIR",
            "HEADER_LIST_DIR", "SAVED_LAYOUT_DIR", "SETTING_EXPORT_DIR",
        ):
            with self.subTest(key=key):
                self.assertGreater(row_of(window.labels[key]), separator_row)
        # It is a directory row like the other roots: open button, tip and all.
        self.assertIn("EMBED_DIR", window.directory_open_buttons)
        for widget in (window.inputs["EMBED_DIR"], window.labels["EMBED_DIR"]):
            from desktop.Tip_Panel import tip_plain

            self.assertIn("$embedding$", tip_plain(window.tip_db[widget]))

    def test_embedding_alias_follows_the_live_embedding_directory(self):
        window = self.window
        original = window.inputs["EMBED_DIR"].text()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                window.inputs["EMBED_DIR"].setText(temp_dir)
                self.assertEqual(window._directory_base_values()["EMBED_DIR"], temp_dir)
                self.assertEqual(
                    pathlib.Path(window._resolved_directory_value("$embedding$")),
                    pathlib.Path(temp_dir),
                )
                self.assertEqual(
                    pathlib.Path(window._resolved_directory_value(r"$embedding$\sub")),
                    pathlib.Path(temp_dir, "sub"),
                )
                self.assertEqual(
                    pathlib.Path(window._absolute_project_path("$embedding$/sub")),
                    pathlib.Path(temp_dir, "sub").resolve(),
                )
            finally:
                window.inputs["EMBED_DIR"].setText(original)
                self.app.processEvents()

    def test_changing_the_embedding_directory_refreshes_its_dropdown_once(self):
        window = self.window
        original = window.inputs["EMBED_DIR"].text()
        original_input = window.inputs["INPUT_FILE_DIR"].text()
        refreshed = []
        real_refresh = window.refresh_combo

        def counting_refresh(combo, key, extensions):
            if combo is window.cb_embed:
                refreshed.append(key)
            return real_refresh(combo, key, extensions)

        with tempfile.TemporaryDirectory() as temp_dir:
            pathlib.Path(temp_dir, "listed_[model]_embeddings.h5").write_bytes(b"")
            try:
                with mock.patch.object(window, "refresh_combo", counting_refresh):
                    window.inputs["EMBED_DIR"].setText(temp_dir)
                self.assertEqual(refreshed, ["EMBED_DIR"])
                self.assertEqual(
                    [window.cb_embed.itemText(i) for i in range(window.cb_embed.count())],
                    ["", "listed_[model]_embeddings.h5"],
                )
                # A folder written under the input root is re-listed when that
                # root changes, as before, and just as often.
                refreshed.clear()
                with mock.patch.object(window, "refresh_combo", counting_refresh):
                    window.inputs["INPUT_FILE_DIR"].setText(temp_dir)
                self.assertEqual(refreshed, ["EMBED_DIR"])
            finally:
                window.inputs["INPUT_FILE_DIR"].setText(original_input)
                window.inputs["EMBED_DIR"].setText(original)
                self.app.processEvents()

    def test_old_settings_holding_embeddings_as_the_embedding_directory_still_load(self):
        namespace = self.namespace
        defaults = namespace["DIRECTORY_PROFILE_DEFAULTS"]
        # Settings and profiles written before the folder became a root hold
        # the same value.
        raw = dict(defaults, EMBED_DIR="Embeddings")
        self.assertEqual(self.window._normalize_profile_data("directories", raw), raw)
        # Applying settings resolves aliases in the module's directory globals;
        # the window shares them, so put every one back.
        with mock.patch.dict(namespace):
            namespace["apply_viewer_settings"]({"EMBED_DIR": "Embeddings"})
            self.assertEqual(namespace["EMBED_DIR"], "Embeddings")
            # So does a folder an old file wrote under the input root.
            stored = os.path.join("$input_file$", "Embeddings")
            namespace["apply_viewer_settings"]({"EMBED_DIR": stored})
            self.assertEqual(
                namespace["EMBED_DIR"], namespace["resolve_directory_path"](stored)
            )
            self.assertNotIn("$", namespace["EMBED_DIR"])

    def test_deprecated_directory_profile_keys_are_ignored_and_pruned(self):
        defaults = self.namespace["DIRECTORY_PROFILE_DEFAULTS"]
        raw = dict(defaults)
        raw.update({key: "legacy/path" for key in self.namespace["DEPRECATED_DIRECTORY_KEYS"]})
        raw["FASTA_DIR"] = r"Input_Files\Sequence_Sets"
        raw["SAVED_LAYOUT_DIR"] = "Cache_Files/Saved_Layouts"
        normalized = self.window._normalize_profile_data("directories", raw)
        self.assertEqual(normalized, defaults)

        raw["UNRELATED_UNKNOWN_KEY"] = "invalid"
        with self.assertRaisesRegex(ValueError, "UNRELATED_UNKNOWN_KEY"):
            self.window._normalize_profile_data("directories", raw)

        method_globals = self.window._read_custom_settings.__func__.__globals__
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = pathlib.Path(temp_dir, "viewer_settings.json")
            settings_path.write_text(json.dumps(raw), encoding="utf-8")
            with mock.patch.dict(
                method_globals, {"DEFAULT_SETTINGS_FILE": str(settings_path)}
            ):
                loaded = self.window._read_custom_settings()
        for key in self.namespace["DEPRECATED_DIRECTORY_KEYS"]:
            self.assertNotIn(key, loaded)
        self.assertEqual(loaded["FASTA_DIR"], defaults["FASTA_DIR"])
        self.assertEqual(loaded["SAVED_LAYOUT_DIR"], defaults["SAVED_LAYOUT_DIR"])
        self.assertIn("UNRELATED_UNKNOWN_KEY", loaded)

    def test_obsolete_layout_engine_profile_keys_are_ignored(self):
        defaults = self.namespace["PHYSICS_PROFILE_DEFAULTS"]
        raw = dict(defaults)
        for key in self.namespace["OBSOLETE_LAYOUT_ENGINE_KEYS"]:
            raw[key] = 123

        normalized = self.window._normalize_profile_data(
            "simulation_physics", raw
        )
        self.assertEqual(normalized, defaults)
        self.assertNotIn("PHYSICS_ENGINE", normalized)
        self.assertFalse(any(key.startswith("MC_") for key in normalized))
        self.assertFalse(any(key.startswith("SGLD_") for key in normalized))

        raw["UNRELATED_UNKNOWN_KEY"] = "invalid"
        with self.assertRaisesRegex(ValueError, "UNRELATED_UNKNOWN_KEY"):
            self.window._normalize_profile_data("simulation_physics", raw)

    def test_all_tabs_share_padding_and_separator_spacing(self):
        from PySide6.QtCore import QPoint

        original_index = self.window.tabs.currentIndex()
        checks = (
            ("inputs_outputs", None, "NODE_FASTA_FILE"),
            ("visual_effects", None, "TEXT_SIZE"),
            ("simulation_physics", None, "LAYOUT_DEVICE_SELECTION"),
            ("directories", "SAVED_CONFIG_DIR", "INPUT_FILE_DIR"),
        )
        expected_margin = self.namespace["CONFIG_TAB_CONTENT_MARGIN"]
        expected_spacing = self.namespace["CONFIG_SEPARATOR_PADDING"]
        expected_thickness = self.namespace["CONFIG_SEPARATOR_THICKNESS"]
        expected_field_x = (
            expected_margin
            + self.window.label_column_width
            + self.namespace["CONFIG_FIELD_HORIZONTAL_SPACING"]
        )
        profile_field_positions = []
        setting_field_positions = []
        self.assertEqual(expected_spacing, 24)
        self.assertEqual(expected_thickness, self.tokens["separator_thickness"])  # the theme's lines
        self.assertEqual(
            self.namespace["DEFAULT_SAVED_CONFIG_DIR"],
            os.path.join("$cache_file$", "Saved_Config"),
        )
        self.assertEqual(
            self.namespace["_migrate_saved_config_dir"]("Saved_Config"),
            os.path.join("$cache_file$", "Saved_Config"),
        )
        self.assertEqual(
            self.namespace["_migrate_saved_config_dir"](
                r"Cache_Files\Saved_Config"
            ),
            os.path.join("$cache_file$", "Saved_Config"),
        )
        self.assertEqual(
            self.namespace["_migrate_saved_config_dir"]("D:/SSN/Profiles"),
            "D:/SSN/Profiles",
        )
        self.assertEqual(
            self.namespace["_resolved_saved_config_root"](
                os.path.join("$cache_file$", "Saved_Config"),
                {"CACHE_FILE_DIR": os.path.join("custom", "cache")},
            ),
            (ROOT / "custom" / "cache" / "Saved_Config").resolve(),
        )

        try:
            for index, (tab_id, preceding_key, first_key) in enumerate(checks):
                self.window.tabs.setCurrentIndex(index)
                self.app.processEvents()
                tab = self.window.tabs.widget(index).widget()
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
        original_tip = self.window.tip_panel.text()
        cases = []
        for tab_id in self.window.profile_selectors:
            cases.extend(
                (
                    (self.window.profile_labels[tab_id], "Selects the settings profile"),
                    (self.window.profile_selectors[tab_id], "Selects the settings profile"),
                    (self.window.profile_name_inputs[tab_id], "Selects the settings profile"),
                    (self.window.profile_folder_buttons[tab_id], "Selects the settings profile"),
                )
            )
        cases.extend(
            (
                (self.window.labels["TARGET_CACHE"], "Compatibility-specific folder"),
                (self.window.lbl_cache_tracker, "Compatibility-specific folder"),
                (self.window.btn_open_cache, "Compatibility-specific folder"),
            )
        )
        try:
            for target, expected_tip in cases:
                with self.subTest(target=type(target).__name__):
                    self.assertIn(target, self.window.tip_db)
                    self.window.tip_panel.setText("sentinel")
                    self.window.eventFilter(target, left_press())
                    self.assertIn(expected_tip, self.window.tip_panel.text())
        finally:
            self.window.tip_panel.setText(original_tip)

    def test_native_tooltip_popup_is_redirected_to_shared_tip_panel(self):
        from PySide6.QtCore import QEvent, QPoint
        from PySide6.QtGui import QHelpEvent

        original_tip = self.window.tip_panel.text()
        try:
            # Hovering replaces only a tip (TipPanelStatusLockTests), so start
            # from one, whatever earlier tests left in the shared window.
            self.window.eventFilter(self.window.labels["NODE_FASTA_FILE"], left_press())
            self.assertIn("Primary FASTA file", self.window.tip_panel.text())
            handled = self.window.eventFilter(
                self.window.btn_open_cache,
                QHelpEvent(
                    QEvent.Type.ToolTip,
                    QPoint(1, 1),
                    QPoint(1, 1),
                ),
            )

            self.assertTrue(handled)
            self.assertIn("Compatibility-specific folder", self.window.tip_panel.text())
        finally:
            self.window.tip_panel.setText(original_tip)

    def test_save_only_reports_success_in_the_tip_panel_without_popup(self):
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

    def test_save_reports_default_tabs_in_the_tip_panel(self):
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

    def test_new_profile_without_name_reports_in_the_tip_panel_without_popup(self):
        globals_dict = self.window.save_settings.__globals__
        original_settings_file = globals_dict["DEFAULT_SETTINGS_FILE"]
        original_root = self.window.inputs["SAVED_CONFIG_DIR"].text()
        original_custom = dict(self.window._custom_settings)
        original_tip = self.window.tip_panel.text()
        selector = self.window.profile_selectors["visual_effects"]
        name_input = self.window.profile_name_inputs["visual_effects"]
        original_selection = selector.currentText()
        original_name = name_input.text()

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Only the name check stops this save, so a regression must
                # write to temp rather than the developer's own settings.
                globals_dict["DEFAULT_SETTINGS_FILE"] = str(
                    pathlib.Path(temp_dir, "viewer_settings.json")
                )
                self.window.inputs["SAVED_CONFIG_DIR"].setText(temp_dir)
                self.window._saved_config_directory_committed()
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
                self.assertEqual(list(pathlib.Path(temp_dir).rglob("*.json")), [])
            finally:
                globals_dict["DEFAULT_SETTINGS_FILE"] = original_settings_file
                self.window._custom_settings = original_custom
                self.window.inputs["SAVED_CONFIG_DIR"].setText(original_root)
                self.window._saved_config_directory_committed()
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
                self.assertIn("Created profile: Visual Effects:", self.window.tip_panel.text())
                self.assertIn("publication", self.window.tip_panel.text())
                self.assertIn(
                    "Created profiles: Visual Effects: 'a', Directories: 'b'.",
                    self.window._save_success_message([("visual_effects", "a"), ("directories", "b")], []),
                )

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


if __name__ == "__main__":
    unittest.main()
