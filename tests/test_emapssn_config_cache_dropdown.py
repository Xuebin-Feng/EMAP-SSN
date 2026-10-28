import os
import json
import pathlib
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import h5py
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _left_press():
    """A left mouse-button press, which always puts an element's tip in the panel."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(1, 1),
        QPointF(1, 1),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


class CacheDropdownRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Acceleration as Hardware_Utils  # noqa: F401 - load torch before PySide6
        from EMAPSSN_Tools import DynamicComboBox
        from PySide6.QtWidgets import QApplication
        from tests.config_gui_loader import load_config_namespace, open_config_window
        from tests.theme_fixture import apply_theme_for_class
        from desktop.Studio_Theme import SWITCH_STYLESHEET, TOKENS

        # Keep the developer's viewer_settings.json, and the inputs it selects,
        # out of both the module defaults and the window.
        namespace = load_config_namespace()
        cls.app = QApplication.instance() or QApplication([])
        namespace["configure_qt_application_fonts"](cls.app)
        apply_theme_for_class(cls, cls.app)
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.window = open_config_window(namespace["ConfigGUI"], cls.temp_directory.name)
        cls.window.show()
        cls.namespace = namespace
        cls.switch_stylesheet = SWITCH_STYLESHEET
        cls.tokens = TOKENS
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
        cls.temp_directory.cleanup()

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

    def test_incompatible_canonical_folder_uses_manifest_suffix(self):
        cache_manifest = self.namespace["cache_manifest"]
        current_compatibility = cache_manifest.build_compatibility(
            "a" * 64,
            "b" * 64,
            "alignment",
            alignment_score="global",
            normalization="alignment_length",
            top_edge_percent=5.0,
        )
        incompatible_compatibility = cache_manifest.build_compatibility(
            "c" * 64,
            "b" * 64,
            "alignment",
            alignment_score="global",
            normalization="alignment_length",
            top_edge_percent=5.0,
        )
        incompatible_manifest = cache_manifest.build_manifest(
            {"basename": "set.fasta", "size_bytes": 1, "sha256": "c" * 64},
            {"basename": "network.h5", "size_bytes": 1, "sha256": "b" * 64},
            incompatible_compatibility,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            canonical = root / "readable-cache"
            cache_manifest.write_manifest_atomic(canonical, incompatible_manifest)
            directory_input = self.window.inputs["SAVED_LAYOUT_DIR"]
            previous_directory = directory_input.text()
            directory_input.blockSignals(True)
            directory_input.setText(str(root))
            directory_input.blockSignals(False)
            try:
                with mock.patch.object(
                    self.window,
                    "_cache_paths_from_inputs",
                    return_value=(str(root / "set.fasta"), str(root / "network.h5")),
                ), mock.patch.object(
                    cache_manifest,
                    "build_compatibility",
                    return_value=current_compatibility,
                ), mock.patch.object(
                    cache_manifest,
                    "build_canonical_cache_name",
                    return_value=canonical.name,
                ), mock.patch.object(
                    cache_manifest,
                    "find_matching_manifest_folders",
                    return_value=[],
                ):
                    self.window._apply_cache_discovery(
                        {
                            "sequence": {"sha256": "a" * 64},
                            "network": {"sha256": "b" * 64},
                            "network_type": "alignment",
                        }
                    )

                suffix = cache_manifest.calculate_manifest_id(
                    current_compatibility
                )[:8]
                expected = root / f"{canonical.name}_[{suffix}]"
                self.assertEqual(
                    pathlib.Path(self.window.current_cache_folder), expected
                )
                self.assertIn(expected.name, self.window.lbl_cache_tracker.text())
                self.assertEqual(
                    cache_manifest.read_manifest(canonical)["manifest_id"],
                    incompatible_manifest["manifest_id"],
                )
            finally:
                directory_input.blockSignals(True)
                directory_input.setText(previous_directory)
                directory_input.blockSignals(False)

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
            self.window._toggle_new_cache_input()
            self.app.processEvents()

            tab = self.window.tabs.widget(0).widget()
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

    def test_new_cache_focus_is_user_initiated_and_reference_typing_stays_put(self):
        from PySide6.QtTest import QTest

        combo = self.window.cb_cache_file
        original_tab = self.window.tabs.currentIndex()
        original_focus = self.app.focusWidget()
        original_items = [
            (combo.itemText(index), combo.itemData(index))
            for index in range(combo.count())
        ]
        original_combo_index = combo.currentIndex()
        original_combo_enabled = combo.isEnabled()
        original_cache_folder = self.window.current_cache_folder
        original_cache_launch_allowed = self.window._cache_launch_allowed
        original_reference = self.window.line_ref.text()
        original_cache_name = self.window.line_new_cache.text()

        try:
            self.window.tabs.setCurrentIndex(0)
            self.window.show()
            self.window.activateWindow()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(New Layout Cache)", None)
            combo.setCurrentIndex(0)
            combo.setEnabled(True)
            combo.blockSignals(False)
            self.window._cache_launch_allowed = True
            self.window._toggle_new_cache_input()
            self.window.line_new_cache.setText("cache-name")

            with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
                self.window, "_request_cache_discovery"
            ) as request_cache_discovery:
                self.window.current_cache_folder = temp_dir
                self.window.line_ref.clear()
                self.window.line_ref.setFocus()
                self.app.processEvents()

                # A programmatic cache refresh may update the conditional field,
                # but it must not redirect keyboard focus to that field.
                self.window._refresh_cache_file_combo()
                self.app.processEvents()
                self.assertIs(self.app.focusWidget(), self.window.line_ref)

                for character in "ref42":
                    QTest.keyClicks(self.app.focusWidget(), character)
                    self.app.processEvents()
                    self.assertIs(self.app.focusWidget(), self.window.line_ref)

                self.assertEqual(self.window.line_ref.text(), "ref42")
                self.assertEqual(self.window.line_new_cache.text(), "cache-name")
                self.assertEqual(combo.currentText(), "(New Layout Cache)")
                self.assertTrue(self.window.spin_alignment_offset.isEnabled())
                self.assertTrue(self.window.lbl_alignment_offset.isEnabled())

                self.window.line_ref.setText("   ")
                self.assertFalse(self.window.spin_alignment_offset.isEnabled())
                self.assertFalse(self.window.lbl_alignment_offset.isEnabled())
                request_cache_discovery.assert_not_called()

                # activated is emitted only for a user selection, so this is
                # the one path that should move focus to the cache-name field.
                combo.activated.emit(combo.currentIndex())
                self.app.processEvents()
                self.assertIs(self.app.focusWidget(), self.window.line_new_cache)
        finally:
            self.window.current_cache_folder = original_cache_folder
            self.window._cache_launch_allowed = original_cache_launch_allowed
            combo.blockSignals(True)
            combo.clear()
            for text, data in original_items:
                combo.addItem(text, data)
            combo.setCurrentIndex(original_combo_index)
            combo.setEnabled(original_combo_enabled)
            combo.blockSignals(False)
            self.window._toggle_new_cache_input()
            if self.window._new_cache_selected():
                self.window.line_new_cache.setText(original_cache_name)
            self.window.line_ref.setText(original_reference)
            self.window.tabs.setCurrentIndex(original_tab)
            if original_focus is not None and original_focus.isVisible():
                original_focus.setFocus()
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
                    self.window.eventFilter(target, _left_press())
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
            self.window.eventFilter(self.window.labels["NODE_FASTA_FILE"], _left_press())
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

    def test_layout_export_contains_only_generation_settings_and_exact_name(self):
        generation_values = {
            "NODE_FASTA_FILE": "Input_Files/Sequence_Sets/example.fasta",
            "INPUT_HDF5": "Input_Files/Networks_EValues/example.h5",
            "ALIGNMENT_SCORE": "global",
            "NORM_MODE": "alignment_length",
            "UMAP_MODE": False,
            "UMAP_NEIGHBORS": 15,
            "UMAP_MIN_DIST": 0.1,
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
                self.window.inputs["SETTING_EXPORT_DIR"],
                "text",
                return_value=str(temp_path / "layout_exports"),
            ), mock.patch.object(
                self.window, "_cache_launch_allowed", True
            ), mock.patch.object(
                self.window, "current_cache_folder", str(temp_path / "target")
            ), mock.patch.dict(
                method_globals, {"PROJECT_ROOT": temp_path}
            ):
                settings = self.window._collect_layout_generation_settings()
                document = settings.to_document(project_root=temp_path)
                from desktop.Viewer_State import decode_document
                payload = decode_document(document, "layout")
                self.assertEqual(document["schema_version"], 2)
                self.assertEqual(document["kind"], "layout")
                self.assertEqual(payload["CACHE_FILENAME"], "exact.h5")
                self.assertIs(payload["UMAP_MODE"], False)
                self.assertIsInstance(payload["MAX_STEPS"], int)
                self.assertNotIn("PHYSICS_ENGINE", payload)
                self.assertFalse(any(key.startswith("MC_") for key in payload))
                self.assertFalse(any(key.startswith("SGLD_") for key in payload))
                self.assertNotIn("NODE_SIZE", payload)
                self.assertNotIn("MSA_FILE", payload)
                self.assertNotIn("PRINT_SAVE_DIR", payload)
                self.assertEqual(payload["SAVED_LAYOUT_DIR"], "layouts")

                with mock.patch.object(
                    self.namespace["QFileDialog"],
                    "getSaveFileName",
                    return_value=(str(target_json), "JSON Files (*.json)"),
                ) as get_save_file_name, mock.patch.object(
                    self.namespace["QMessageBox"], "information"
                ) as information, mock.patch.object(
                    self.namespace["QMessageBox"], "critical"
                ) as critical:
                    self.window.export_layout_settings()
                    suggested_path = pathlib.Path(
                        get_save_file_name.call_args.args[2]
                    )
                    self.assertEqual(
                        suggested_path.parent,
                        temp_path / "layout_exports",
                    )

            exported = json.loads(target_json.read_text(encoding="utf-8"))
            self.assertEqual(exported, document)
            information.assert_called_once()
            self.assertIn(
                "Layout_Cache_Generator.py",
                information.call_args.args[2],
            )
            critical.assert_not_called()

    def test_layout_export_button_is_enabled_only_for_new_cache_generation(self):
        # Save & Run is the one primary action; Export is an outline button.
        self.assertEqual(self.window.btn_save_run.property("role"), "primary")
        self.assertIsNone(self.window.btn_export_layout.property("role"))
        self.assertEqual(self.window.btn_export_layout.styleSheet(), "")
        self.assertEqual(
            self.window.btn_export_layout.sizeHint().height(),
            self.window.btn_save_run.sizeHint().height(),
        )
        self.assertEqual(
            self.window.btn_export_layout.sizeHint().height(),
            self.window.btn_check.sizeHint().height(),
        )
        combo = self.window.cb_cache_file
        original_items = [
            (combo.itemText(index), combo.itemData(index))
            for index in range(combo.count())
        ]
        original_index = combo.currentIndex()
        try:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("version_00.h5", "layout/version_00.h5")
            combo.addItem("(New Layout Cache)", None)
            combo.blockSignals(False)
            with mock.patch.object(self.window, "_cache_launch_allowed", True):
                combo.setCurrentIndex(1)
                self.assertTrue(self.window.btn_export_layout.isEnabled())
                combo.setCurrentIndex(0)
                self.assertFalse(self.window.btn_export_layout.isEnabled())
        finally:
            combo.blockSignals(True)
            combo.clear()
            for text, data in original_items:
                combo.addItem(text, data)
            combo.setCurrentIndex(original_index)
            combo.blockSignals(False)
            self.window._toggle_new_cache_input()


class CacheDiscoveryRaceTests(unittest.TestCase):
    """Hashing left running for earlier inputs never decides the target cache.

    Discovery hashes the selected FASTA and network in a CacheHashWorker,
    unless both hashes are cached already; then it applies them at once. That
    synchronous path used to leave the running worker's request current, so
    the worker's late result was cached under the newly selected files and
    shown for them. Here each worker hashes only when a test finishes it, on
    the test's thread, so every interleaving is deterministic.

    Nor does a check that ends in Save & Run's pause after a click, which
    keeps a double click from launching twice, cut the pause short or outlast it.

    A cache published after the check, as the Layout Generator publishes one
    (manifest, then .h5), shows at the window's next look without the inputs
    being chosen again; a look that finds what is shown changes nothing.
    """

    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Acceleration as Hardware_Utils  # noqa: F401 - load torch before PySide6
        from PySide6.QtWidgets import QApplication
        from tests.config_gui_loader import load_config_namespace

        cls.namespace = load_config_namespace()
        cls.cache_manifest = cls.namespace["cache_manifest"]
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from tests.config_gui_loader import open_config_window

        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        root = pathlib.Path(home.name)
        self._defer_hashing()
        self.window = open_config_window(self.namespace["ConfigGUI"], root)
        self.addCleanup(self._close_window)

        # Two FASTA files share one BLAST network, and each pair has its own
        # compatible layout folder, so the tracker names the hashes it used.
        self.work = root / "work"
        self.work.mkdir()
        (self.work / "a.fasta").write_text(">a1\nMKTAYIAK\n>a2\nMKTAYIAR\n", encoding="utf-8")
        (self.work / "b.fasta").write_text(">b1\nMSDNELKQ\n>b2\nMSDNELKE\n", encoding="utf-8")
        with h5py.File(self.work / "network.h5", "w") as network:
            network.attrs["model_name"] = "BLAST"
            network.create_dataset("headers", data=[b"a1", b"a2"])
            network.create_dataset("i", data=np.asarray([0], dtype=np.uint16))
            network.create_dataset("j", data=np.asarray([1], dtype=np.uint16))
            network.create_dataset("score", data=np.asarray([1e-30]))
        layouts = root / "layouts"
        self.layout_folders = {
            name: self._write_layout_folder(layouts, name)
            for name in ("a.fasta", "b.fasta")
        }
        self.window.inputs["SAVED_LAYOUT_DIR"].setText(str(layouts))
        self.window.inputs["FASTA_DIR"].setText(str(self.work))
        self.window.inputs["HDF5_DIR"].setText(str(self.work))
        self.window.cb_hdf5.setCurrentText("network.h5")

    def _defer_hashing(self):
        """Replace CacheHashWorker with one that hashes only when _finish() runs it."""
        self.workers, self.interrupted = [], set()
        workers, interrupted = self.workers, self.interrupted

        class DeferredHashWorker(self.namespace["CacheHashWorker"]):
            def start(self):
                workers.append(self)

            # A thread that never started ignores requestInterruption(), so
            # keep the request where run() looks for it.
            def requestInterruption(self):
                interrupted.add(self.request_id)

            def isInterruptionRequested(self):
                return self.request_id in interrupted

        patcher = mock.patch.dict(self.namespace, {"CacheHashWorker": DeferredHashWorker})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_layout_folder(self, layouts, fasta_name, folder_name=None):
        """Write a layout folder whose manifest matches this FASTA with the network."""
        sequence = self.cache_manifest.fingerprint_file(self.work / fasta_name)
        network = self.cache_manifest.fingerprint_file(self.work / "network.h5")
        compatibility = self.cache_manifest.build_compatibility(
            sequence["sha256"],
            network["sha256"],
            "blast",
            **self.window._cache_setting_values(),
        )
        folder = layouts / (folder_name or f"{pathlib.Path(fasta_name).stem}_layout")
        self.cache_manifest.write_manifest_atomic(
            folder, self.cache_manifest.build_manifest(sequence, network, compatibility)
        )
        return folder

    def _close_window(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def _select(self, fasta_name):
        self.window.cb_fasta.setCurrentText(fasta_name)
        self.assertEqual(self.window.cb_fasta.currentText(), fasta_name)

    def _finish(self, worker):
        """Hash and report on this thread, as the worker's own thread would."""
        worker.run()
        self.app.processEvents()

    def _tracker(self):
        return self.window.lbl_cache_tracker.text()

    def _shown(self, fasta_name):
        return f"Compatible Folder: {self.layout_folders[fasta_name].name}"

    def _items(self):
        combo = self.window.cb_cache_file
        return [combo.itemText(index) for index in range(combo.count())]

    def _needs_computing(self, fasta_name):
        """Select a FASTA whose layout folder is gone; return the folder the window targets."""
        shutil.rmtree(self.layout_folders[fasta_name])
        self._select(fasta_name)
        self._finish(self.workers[-1])
        target = pathlib.Path(self.window.current_cache_folder)
        self.assertEqual(self._tracker(), f"Target Folder: {target.name} [Needs Computing]")
        self.assertEqual(self._items(), ["(New Layout Cache)"])
        return target

    def _generate(self, fasta_name, folder, cache_name="version_00.h5"):
        """Publish a cache as the Layout Generator does: the manifest, then the .h5 file."""
        if not (folder / self.cache_manifest.MANIFEST_FILENAME).exists():
            self._write_layout_folder(folder.parent, fasta_name, folder.name)
        (folder / cache_name).write_bytes(b"")

    def test_hashing_left_running_for_earlier_inputs_never_overrides_cached_ones(self):
        """a.fasta is still being hashed when b.fasta, already hashed, is chosen again."""
        self._select("b.fasta")
        self._finish(self.workers[-1])
        self.assertEqual(self._tracker(), self._shown("b.fasta"))

        self._select("a.fasta")
        hashing_a = self.workers[-1]
        request_a = hashing_a.request_id
        self.assertEqual(self._tracker(), "Checking input files…")
        self._select("b.fasta")
        self.assertEqual(self._tracker(), self._shown("b.fasta"))

        self._finish(hashing_a)
        self.assertEqual(self._tracker(), self._shown("b.fasta"))
        self.assertEqual(
            pathlib.Path(self.window.current_cache_folder),
            self.layout_folders["b.fasta"],
        )
        # Nor was a.fasta's hash cached for b.fasta: the next refresh agrees.
        self.window.update_live_validators()
        self.assertEqual(self._tracker(), self._shown("b.fasta"))
        # The worker was told to stop, not left to hash for nothing.
        self.assertIn(request_a, self.interrupted)

    def test_inputs_whose_hashing_was_dropped_are_hashed_again(self):
        self._select("b.fasta")
        self._finish(self.workers[-1])
        self._select("a.fasta")
        dropped = self.workers[-1]
        self._select("b.fasta")
        self._select("a.fasta")
        # The dropped worker's result will be ignored, so a.fasta needs another.
        self.assertEqual(self._tracker(), "Checking input files…")
        self.assertEqual(len(self.workers), 3)

        self._finish(dropped)
        self.assertEqual(self._tracker(), "Checking input files…")
        self._finish(self.workers[-1])
        self.assertEqual(self._tracker(), self._shown("a.fasta"))

    def test_a_file_rewritten_while_it_is_hashed_is_hashed_again(self):
        """Hashes are cached for the files as they were when hashing began,
        never for whatever is on disk when the result arrives."""
        a_fasta = self.work / "a.fasta"
        fingerprint_file = self.cache_manifest.fingerprint_file

        def fingerprint_then_rewrite(path, **options):
            record = fingerprint_file(path, **options)
            if os.path.basename(path) == "a.fasta":
                # Replaced right after it was read: it now holds b.fasta's sequences.
                a_fasta.write_bytes((self.work / "b.fasta").read_bytes())
                modified = a_fasta.stat().st_mtime_ns + 5_000_000_000
                os.utime(a_fasta, ns=(modified, modified))
            return record

        self._select("a.fasta")
        with mock.patch.object(
            self.cache_manifest, "fingerprint_file", side_effect=fingerprint_then_rewrite
        ):
            self._finish(self.workers[-1])
        self.window.update_live_validators()  # as the next change in the window would
        self.assertEqual(self._tracker(), "Checking input files…")
        self.assertEqual(len(self.workers), 2)
        self._finish(self.workers[-1])
        self.assertEqual(self._tracker(), self._shown("b.fasta"))

    def test_giving_up_on_the_selected_inputs_stops_their_hashing(self):
        """However discovery gives up on the selected inputs, it stops the
        worker hashing them and ignores that worker's result."""
        window = self.window

        def clear_the_selection():
            window.cb_fasta.setCurrentText("")

        def fail_to_read_the_inputs():
            with mock.patch.object(
                self.cache_manifest, "file_cache_key", side_effect=OSError("access denied")
            ):
                window.update_live_validators()

        def choose_inputs_that_need_hashing():
            window.cb_fasta.setCurrentText("b.fasta")

        def remove_the_network():
            (self.work / "network.h5").unlink()
            window.update_live_validators()

        for fasta_name, give_up, message in (
            ("a.fasta", clear_the_selection, "Target Cache: Missing FASTA or HDF5"),
            ("b.fasta", fail_to_read_the_inputs, "Cache input error: access denied"),
            ("a.fasta", choose_inputs_that_need_hashing, "Checking input files…"),
            ("a.fasta", remove_the_network, "Target Cache: Selected input file is missing"),
        ):
            with self.subTest(give_up.__name__):
                self._select(fasta_name)
                hashing = self.workers[-1]
                give_up()
                self.assertIn(hashing.request_id, self.interrupted)
                self._finish(hashing)
                self.assertEqual(self._tracker(), message)

    def test_a_double_click_on_save_and_run_launches_once_and_the_button_comes_back(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        self._select("a.fasta")
        self._finish(self.workers[-1])
        self.window.show()
        button = self.window.btn_save_run
        self.assertTrue(button.isEnabled())

        self.assertEqual(self.namespace["CONFIG_RUN_BUTTON_PAUSE_MS"], 1000)
        with mock.patch.object(self.window, "save_and_run") as save_and_run, \
                mock.patch.dict(self.namespace, {"CONFIG_RUN_BUTTON_PAUSE_MS": 150}):
            started = time.monotonic()
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            save_and_run.assert_called_once_with()
            self.assertFalse(button.isEnabled())
            deadline = started + 5.0
            while not button.isEnabled() and time.monotonic() < deadline:
                QTest.qWait(10)
            self.assertTrue(button.isEnabled())
            self.assertGreaterEqual(time.monotonic() - started, 0.14)
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            self.assertEqual(save_and_run.call_count, 2)

    def test_a_cache_check_ending_in_the_pause_neither_ends_it_nor_outlasts_it(self):
        self._select("a.fasta")
        self._finish(self.workers[-1])
        button = self.window.btn_save_run
        timers = []
        # The pause ends when the test says, not on a clock.
        stand_in = SimpleNamespace(singleShot=lambda *args: timers.append(args))
        with mock.patch.object(self.window, "save_and_run"), \
                mock.patch.dict(self.namespace, {"QTimer": stand_in}):
            button.click()
            [(pause, context, end_pause)] = timers
            self.assertEqual((pause, context), (1000, button))
            self.assertFalse(button.isEnabled())

            # b.fasta is hashed, and its cache found, before the pause ends.
            self._select("b.fasta")
            self._finish(self.workers[-1])
            self.assertEqual(self._tracker(), self._shown("b.fasta"))
            self.assertFalse(button.isEnabled())
            end_pause()
            self.assertTrue(button.isEnabled())

            # The inputs are cleared in the next pause: the button stays grey after it.
            button.click()
            end_pause = timers[-1][2]
            self._select("")
            self.assertEqual(self._tracker(), "Target Cache: Missing FASTA or HDF5")
            end_pause()
            self.assertFalse(button.isEnabled())

    def test_a_cache_generated_after_the_check_shows_at_the_next_look(self):
        target = self._needs_computing("a.fasta")
        needs_computing = self._tracker()
        self.window._recheck_saved_caches()  # nothing published yet
        self.assertEqual(self._tracker(), needs_computing)

        self._generate("a.fasta", target)
        self.window._recheck_saved_caches()
        self.assertEqual(self._tracker(), f"Compatible Folder: {target.name}")
        self.assertEqual(self._items(), ["version_00.h5", "(New Layout Cache)"])
        # Selected, so Save & Run opens it rather than computing another.
        self.assertEqual(self.window.cb_cache_file.currentText(), "version_00.h5")
        self.assertTrue(self.window._cache_launch_allowed)
        self.assertEqual(len(self.workers), 1, "the inputs were not hashed again")

    def test_a_manifest_published_before_its_cache_shows_the_folder_then_the_cache(self):
        target = self._needs_computing("a.fasta")
        self._write_layout_folder(target.parent, "a.fasta", target.name)
        self.window._recheck_saved_caches()
        self.assertEqual(self._tracker(), f"Compatible Folder: {target.name}")
        self.assertEqual(self._items(), ["(New Layout Cache)"])

        # Still being written: the Layout Generator's .partial file is no cache yet.
        (target / ".version_00.h5.x1y2.partial").write_bytes(b"")
        with mock.patch.object(self.window, "_request_cache_discovery") as search:
            self.window._recheck_saved_caches()
        search.assert_not_called()

        self._generate("a.fasta", target)
        self.window._recheck_saved_caches()
        self.assertEqual(self.window.cb_cache_file.currentText(), "version_00.h5")

    def test_a_cache_generated_into_the_compatible_folder_is_selected(self):
        self._select("b.fasta")
        self._finish(self.workers[-1])
        self.assertEqual(self._items(), ["(New Layout Cache)"])
        self.window.line_new_cache.setText("mine.h5")

        self._generate("b.fasta", self.layout_folders["b.fasta"], "mine.h5")
        self.window._recheck_saved_caches()
        self.assertEqual(self._items(), ["mine.h5", "(New Layout Cache)"])
        self.assertEqual(self.window.cb_cache_file.currentText(), "mine.h5")
        self.assertEqual(self.window.line_new_cache.text(), "")

    def test_a_look_that_finds_no_new_cache_keeps_the_choice(self):
        folder = self.layout_folders["b.fasta"]
        for age, name in enumerate(("newer.h5", "older.h5")):
            (folder / name).write_bytes(b"")
            os.utime(folder / name, (1_700_000_000 - age, 1_700_000_000 - age))
        self._select("b.fasta")
        self._finish(self.workers[-1])
        combo = self.window.cb_cache_file
        self.assertEqual(self._items(), ["newer.h5", "older.h5", "(New Layout Cache)"])
        combo.setCurrentIndex(combo.findText("older.h5"))

        # A layout folder for other inputs appears: the window is left alone.
        self._write_layout_folder(folder.parent, "a.fasta", "other_inputs_layout")
        with mock.patch.object(combo, "clear", wraps=combo.clear) as clear:
            self.window._recheck_saved_caches()
        clear.assert_not_called()
        self.assertEqual(combo.currentText(), "older.h5")

        # A cache that goes is dropped from the list; the one chosen stays chosen.
        (folder / "newer.h5").unlink()
        self.window._recheck_saved_caches()
        self.assertEqual(self._items(), ["older.h5", "(New Layout Cache)"])
        self.assertEqual(combo.currentText(), "older.h5")

        # So does a new cache being named.
        combo.setCurrentIndex(combo.count() - 1)
        self.window._toggle_new_cache_input()
        self.window.line_new_cache.setText("draft.h5")
        (folder / "older.h5").unlink()
        self.window._recheck_saved_caches()
        self.assertEqual(self._items(), ["(New Layout Cache)"])
        self.assertEqual(self.window.line_new_cache.text(), "draft.h5")

    def test_a_second_compatible_folder_shows_the_duplicate_error_at_the_next_look(self):
        self._select("b.fasta")
        self._finish(self.workers[-1])
        folder = self.layout_folders["b.fasta"]
        shutil.copytree(folder, folder.with_name("b_layout_copy"))
        self.window._recheck_saved_caches()
        self.assertEqual(self._tracker(), "Error: 2 compatible cache folders found")
        self.assertFalse(self.window._cache_launch_allowed)

        shutil.rmtree(folder.with_name("b_layout_copy"))
        self.window._recheck_saved_caches()
        self.assertEqual(self._tracker(), self._shown("b.fasta"))

    def test_a_cache_published_during_the_search_shows_at_the_next_look(self):
        """What a search looks at is read before it runs, so a cache it missed
        still counts as a change at the next look."""
        target = self._needs_computing("a.fasta")
        find_matching = self.cache_manifest.find_matching_manifest_folders

        def search_then_generate(saved_layout_dir, compatibility):
            matches = find_matching(saved_layout_dir, compatibility)
            self._generate("a.fasta", target)
            return matches

        (target.parent / "unrelated").mkdir()  # so the next look searches
        with mock.patch.object(
            self.cache_manifest, "find_matching_manifest_folders", side_effect=search_then_generate
        ):
            self.window._recheck_saved_caches()
        self.assertIn("[Needs Computing]", self._tracker())
        self.window._recheck_saved_caches()
        self.assertEqual(self.window.cb_cache_file.currentText(), "version_00.h5")

    def test_opening_the_cache_list_looks_again(self):
        target = self._needs_computing("a.fasta")
        self._generate("a.fasta", target)
        self.window.show()
        combo = self.window.cb_cache_file
        combo.showPopup()
        combo.hidePopup()
        self.assertEqual(self._tracker(), f"Compatible Folder: {target.name}")
        self.assertEqual(combo.currentText(), "version_00.h5")

    def test_a_shown_window_looks_again_by_itself(self):
        from PySide6.QtTest import QTest

        self.assertEqual(self.namespace["CONFIG_CACHE_RECHECK_MS"], 1000)
        timer = self.window._cache_recheck_timer
        self.assertTrue(timer.isActive())
        self.assertEqual(timer.interval(), 1000)
        target = self._needs_computing("a.fasta")
        self._generate("a.fasta", target)

        # Hidden, it does not look.
        self.window._recheck_saved_caches_while_shown()
        self.assertIn("[Needs Computing]", self._tracker())

        with mock.patch.dict(self.namespace, {"CONFIG_CACHE_RECHECK_MS": 20}):
            timer.setInterval(20)
            self.window.show()
            deadline = time.monotonic() + 5.0
            while "[Needs Computing]" in self._tracker() and time.monotonic() < deadline:
                QTest.qWait(10)
        self.assertEqual(self._tracker(), f"Compatible Folder: {target.name}")

    def test_a_slow_layout_folder_is_looked_at_less_often(self):
        timer = self.window._cache_recheck_timer
        self.window.show()
        clock = iter([10.0, 10.2, 20.0, 20.001])  # a 200 ms look, then a 1 ms one
        with mock.patch.dict(self.namespace, {"time": SimpleNamespace(perf_counter=lambda: next(clock))}), \
                mock.patch.object(self.window, "_recheck_saved_caches"):
            self.window._recheck_saved_caches_while_shown()
            self.assertEqual(timer.interval(), 4000)
            self.window._recheck_saved_caches_while_shown()
            self.assertEqual(timer.interval(), 1000)

    def test_a_folder_removed_before_its_caches_are_listed_shows_an_error(self):
        self._select("b.fasta")
        self._finish(self.workers[-1])
        self.assertIsNotNone(self.window._saved_cache_look)
        gone = str(self.layout_folders["b.fasta"].parent / "removed_layout")
        with mock.patch.object(
            self.cache_manifest, "find_matching_manifest_folders",
            return_value=[{"folder": gone, "manifest": {}}],
        ):
            self.window.update_live_validators()  # b.fasta's hashes are cached
            self.assertTrue(
                self._tracker().startswith("Cache compatibility error: "), self._tracker()
            )
            self.assertIsNone(self.window._saved_cache_look, "nothing left to look at again")

            # Nor does it escape the result of hashing, which nothing else guards.
            self._select("a.fasta")
            self._finish(self.workers[-1])
            self.assertTrue(
                self._tracker().startswith("Cache compatibility error: "), self._tracker()
            )


class TipPanelStatusLockTests(unittest.TestCase):
    """Hovering never replaces a status message in the tip panel.

    The panel shows both tips and status messages, such as a consistency-check
    result. Hovering (Qt's ToolTip event, after its wake-up delay) shows tips
    only while the panel holds a tip. A left click, or focus by Tab, Shift+Tab
    or a shortcut, on an element with a tip always shows it, and then hovering
    works again. Focus that Qt moves by itself and right clicks count as hovering.
    """

    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Acceleration as Hardware_Utils  # noqa: F401 - load torch before PySide6
        from PySide6.QtWidgets import QApplication
        from tests.config_gui_loader import load_config_namespace

        cls.namespace = load_config_namespace()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from tests.config_gui_loader import open_config_window

        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.root = pathlib.Path(home.name)
        self.window = open_config_window(self.namespace["ConfigGUI"], self.root)
        self.addCleanup(self._close_window)
        self.window.show()
        self.app.processEvents()
        self.labels = self.window.labels
        self.tips = self.window.tip_db_keys

    def _close_window(self):
        self.window.close()
        self.app.processEvents()

    def panel(self):
        from PySide6.QtGui import QTextDocumentFragment
        text = QTextDocumentFragment.fromHtml(self.window.tip_panel.text()).toPlainText()
        # A status message's icon is an image: one object-replacement character.
        return text.replace("\ufffc", "").strip()

    def assert_panel_shows_tip(self, key):
        """The panel shows the tip of key: a plain tip as it is, a rich tip with every text it holds."""
        from desktop.Tip_Panel import Tip, plain

        tip = self.tips[key]
        if not isinstance(tip, Tip):
            self.assertEqual(self.panel(), tip)
            return
        shown = self.panel()

        def squashed(text):
            # A chip pads its text with spaces, which are not in the tip's own text.
            return "".join(text.split())

        for text in tip.texts():
            self.assertIn(squashed(plain(text)), squashed(shown))
        # The title is the field's own label, without its colon.
        self.assertTrue(shown.startswith(self.labels[key].text().rstrip(":")), shown)

    def hover(self, widget):
        from PySide6.QtCore import QEvent, QPoint
        from PySide6.QtGui import QHelpEvent
        from PySide6.QtWidgets import QToolTip
        self.app.sendEvent(widget, QEvent(QEvent.Type.Enter))
        event = QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), widget.mapToGlobal(QPoint(1, 1)))
        self.app.sendEvent(widget, event)
        # Tips go to the panel, never to Qt's popup, even while hovering
        # leaves the panel alone (the Clear button has a tooltip of its own).
        visible = QToolTip.isVisible()
        QToolTip.hideText()
        self.assertFalse(visible)

    def click(self, widget, button=None):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        QTest.mouseClick(widget, button or Qt.MouseButton.LeftButton)

    def focus(self, widget, reason):
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QFocusEvent
        self.app.sendEvent(widget, QFocusEvent(QEvent.Type.FocusIn, reason))

    def run_consistency_check(self):
        """Run the real check on a FASTA that the network covers."""
        window = self.window
        work = self.root / "work"
        work.mkdir()
        (work / "nodes.fasta").write_text(">a\nMKTAYIAK\n>b\nMSEQNNTE\n", encoding="utf-8")
        with h5py.File(work / "network.h5", "w") as network:
            network.create_dataset("headers", data=[b"a", b"b"])
        for combo, name in ((window.cb_fasta, "nodes.fasta"), (window.cb_hdf5, "network.h5")):
            combo.blockSignals(True)
            combo.addItem(name)
            combo.setCurrentText(name)
            combo.blockSignals(False)
        window.cb_msa.blockSignals(True)
        window.cb_msa.clear()
        window.cb_msa.blockSignals(False)
        with mock.patch.object(window, "_resolved_directory_input", return_value=str(work)):
            window.run_consistency_check()
        result = self.panel()
        # The report is a table of rich text, not a plain message.
        self.assertIn("FASTA vs HDF5", result)
        self.assertIn("strict subset", result)
        self.assertIn("<table", self.window.tip_panel.text())
        return result

    def save(self):
        with mock.patch.object(
            self.window,
            "_prepare_profile_writes",
            return_value=([], dict(self.window._custom_settings), [], []),
        ):
            self.window.save_only()
        result = self.panel()
        self.assertTrue(result.startswith("Settings saved successfully."), result)
        return result

    def fail_statistics(self):
        from tests.config_gui_loader import finish_statistics

        with mock.patch("h5py.File", side_effect=OSError("unreadable network")):
            self.window.run_statistics()
            finish_statistics(self.window)
        result = self.panel()
        self.assertEqual(
            result, "Error during network statistics calculation: unreadable network"
        )
        return result

    def test_hovering_shows_tips_while_the_panel_holds_a_tip(self):
        from PySide6.QtCore import QEvent

        placeholder = "Click or tab to an input or its label to see helpful tips here."
        self.assertEqual(self.panel(), placeholder)
        # Unlike Tools, Config waits for Qt's tooltip delay: entering is not enough.
        self.app.sendEvent(self.labels["ALIGNMENT_REFERENCE"], QEvent(QEvent.Type.Enter))
        self.assertEqual(self.panel(), placeholder)
        self.hover(self.labels["ALIGNMENT_REFERENCE"])
        self.assert_panel_shows_tip("ALIGNMENT_REFERENCE")
        self.hover(self.labels["ALIGNMENT_OFFSET"])
        self.assert_panel_shows_tip("ALIGNMENT_OFFSET")
        self.hover(self.window.btn_clear_top_edge)
        # The Clear button's own tooltip replaces the offset tip in the panel.
        clear_tip = self.window.btn_clear_top_edge.toolTip()
        self.assertTrue(clear_tip)
        self.assertEqual(self.panel(), clear_tip)

    def test_a_consistency_check_result_stays_until_a_left_click_on_a_tip(self):
        from PySide6.QtCore import Qt

        result = self.run_consistency_check()
        reference_label = self.labels["ALIGNMENT_REFERENCE"]
        self.hover(reference_label)
        self.hover(self.window.line_ref)
        self.hover(self.window.btn_clear_top_edge)
        self.click(reference_label, Qt.MouseButton.RightButton)
        for reason in (
            Qt.FocusReason.ActiveWindowFocusReason,
            Qt.FocusReason.PopupFocusReason,
            Qt.FocusReason.MouseFocusReason,
            Qt.FocusReason.OtherFocusReason,
        ):
            self.focus(self.window.line_ref, reason)
        self.assertEqual(self.panel(), result)

        self.click(reference_label)
        self.assert_panel_shows_tip("ALIGNMENT_REFERENCE")
        self.hover(self.labels["ALIGNMENT_OFFSET"])
        self.assert_panel_shows_tip("ALIGNMENT_OFFSET")

    def test_tab_shift_tab_and_shortcut_focus_show_the_tip_and_reenable_hovering(self):
        from PySide6.QtCore import Qt

        for reason in (
            Qt.FocusReason.TabFocusReason,
            Qt.FocusReason.BacktabFocusReason,
            Qt.FocusReason.ShortcutFocusReason,
        ):
            with self.subTest(reason=reason.name):
                self.save()
                self.focus(self.window.line_ref, reason)
                self.assert_panel_shows_tip("ALIGNMENT_REFERENCE")
                self.hover(self.labels["ALIGNMENT_OFFSET"])
                self.assert_panel_shows_tip("ALIGNMENT_OFFSET")

    def test_statistics_and_save_messages_also_stay_while_hovering(self):
        for show_status in (self.fail_statistics, self.save):
            with self.subTest(show_status.__name__):
                result = show_status()
                self.hover(self.labels["ALIGNMENT_OFFSET"])
                self.hover(self.window.btn_clear_top_edge)
                self.assertEqual(self.panel(), result)
                self.click(self.labels["ALIGNMENT_REFERENCE"])
                self.assert_panel_shows_tip("ALIGNMENT_REFERENCE")


if __name__ == "__main__":
    unittest.main()
