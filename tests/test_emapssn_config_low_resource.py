# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Config's Low Resource Mode section (EMAPSSN_Config.py, Visual Effects tab).

Under the same padded line as the saved-config row and a "Low Resource Mode"
heading, a switch row per category (Fewer Edges Drawn While Moving, Basic
Animations, Node and Edge Drawing, Webserver Communication) and Show FPS, in
the label column, all off by default. These tests check the section's place
and spacing, the rows and their keys, saving and loading them, the old single
switch in a settings file or saved profile, the tips, and a narrow window.
The settings themselves are in test_low_resource_mode.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402
from desktop.Viewer_State import (  # noqa: E402
    LEGACY_LOW_RESOURCE_KEY, LOW_RESOURCE_DEFAULTS, VISUAL_PROFILE_DEFAULTS, migrate_low_resource_mode,
)
from tests.theme_fixture import apply_theme_for_class  # noqa: E402

SWITCHES = ("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_ANIMATIONS", "LOW_RESOURCE_DRAWING",
            "LOW_RESOURCE_WEBSERVER", "SHOW_FPS")
OLD_CATEGORIES = tuple(LOW_RESOURCE_DEFAULTS)   # what LOW_RESOURCE_MODE stood for



class ConfigSectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Acceleration  # noqa: F401 - torch before PySide6
        from tests.config_gui_loader import load_config_namespace, open_config_window

        cls.namespace = load_config_namespace()
        cls.app = QApplication.instance() or QApplication([])
        cls.namespace["configure_qt_application_fonts"](cls.app)
        apply_theme_for_class(cls, cls.app)
        cls.temp = tempfile.TemporaryDirectory()
        cls.window = open_config_window(cls.namespace["ConfigGUI"], cls.temp.name)
        cls.window._cache_hash_request_id += 1
        for worker in cls.window._cache_hash_workers.values():
            worker.requestInterruption()
            worker.wait()
        cls.window._cache_hash_workers.clear()
        cls.window.resize(1300, 900)
        cls.window.show()
        cls.window.tabs.setCurrentIndex(1)
        cls.app.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls.app.processEvents()
        cls.temp.cleanup()

    def section(self):
        return self.window.findChild(QWidget, "lowResourceSection")

    def rows(self):
        layout = self.section().layout()
        return [layout.itemAt(i).widget() for i in range(1, layout.count())]

    def test_the_section_sits_under_a_line_like_the_saved_config_one(self):
        from PySide6.QtWidgets import QFrame, QLabel
        separator = self.window.findChild(QFrame, "lowResourceSeparator")
        saved_config = self.window.profile_separators["visual_effects"]
        self.assertIs(separator, self.window.low_resource_separator)
        self.assertEqual(separator.property("role"), "separator")
        self.assertEqual(separator.height(), saved_config.height())
        self.assertEqual(separator.parentWidget().layout().contentsMargins(),
                         saved_config.parentWidget().layout().contentsMargins())
        # Under the last colour row, over the heading.
        title = self.window.findChild(QLabel, "lowResourceTitle")
        self.assertEqual(title.text(), "Low Resource Mode")
        self.assertEqual(title.property("role"), "title")
        top = lambda widget: widget.mapTo(self.window, widget.rect().topLeft()).y()  # noqa: E731
        colour = self.window.inputs["NODE_BOUNDARY_COLOR"]
        self.assertLess(top(colour), top(separator))
        self.assertLess(top(separator), top(title))
        self.assertIs(self.section().layout().itemAt(0).widget(), title)
        # The rows above and below keep the saved-config line's distances to it.
        def gaps(wrapper):
            layout = wrapper.parentWidget().layout()
            index = layout.indexOf(wrapper)
            above, below = layout.itemAt(index - 1).geometry(), layout.itemAt(index + 1).geometry()
            line = wrapper.geometry()
            return line.top() - above.bottom(), below.top() - line.bottom()
        self.assertEqual(gaps(separator.parentWidget()), gaps(saved_config.parentWidget()))

    def test_a_switch_row_per_setting_in_the_users_order(self):
        from desktop.Desktop_App import ToggleSwitch
        rows = self.rows()
        self.assertEqual([row.property("settingKey") for row in rows], list(SWITCHES))
        labels = ["Fewer Edges Drawn While Moving:", "Basic Animations:", "Node and Edge Drawing:",
                  "Webserver Communication:", "Show FPS:"]
        field_x = self.window.inputs["NODE_SIZE"].parentWidget().mapTo(self.window, self.window.inputs[
            "NODE_SIZE"].parentWidget().rect().topLeft()).x()
        for row, key, text in zip(rows, SWITCHES, labels):
            with self.subTest(key=key):
                switch, label = self.window.inputs[key], self.window.labels[key]
                self.assertIsInstance(switch, ToggleSwitch)
                self.assertIs(switch.parentWidget(), row)
                self.assertEqual(label.text(), text)
                # In the label column, so the switch starts where the fields above do.
                self.assertEqual(label.width(), self.window.label_column_width)
                self.assertGreaterEqual(label.width(), label.sizeHint().width())
                self.assertEqual(switch.mapTo(self.window, switch.rect().topLeft()).x(), field_x)
                self.assertIn(key, self.window.tip_db_keys)
        # The column fits the longest of them. (Measured on a fresh label: a
        # fixed-width one hints its fixed width.)
        from PySide6.QtWidgets import QLabel
        for key in SWITCHES:
            fresh = QLabel(self.window.labels[key].text())
            fresh.setFont(self.window.labels[key].font())
            with self.subTest(key=key):
                self.assertLessEqual(fresh.sizeHint().width(), self.window.label_column_width)
            fresh.deleteLater()

    def test_all_off_by_default_and_saved_as_the_categories(self):
        window = self.window
        for key in SWITCHES:
            self.assertFalse(window.inputs[key].isChecked(), key)
        chosen = {"FEWER_EDGES_WHILE_MOVING": True, "LOW_RESOURCE_ANIMATIONS": False,
                  "LOW_RESOURCE_DRAWING": True, "LOW_RESOURCE_WEBSERVER": False, "SHOW_FPS": True}
        for key, value in chosen.items():
            window.inputs[key].setChecked(value)
        try:
            for key, value in chosen.items():
                self.assertIs(window.collect_data()[key], value)
            custom = dict(window._custom_settings, **{LEGACY_LOW_RESOURCE_KEY: True})
            with mock.patch.object(window, "_custom_settings", migrate_low_resource_mode(custom)):
                writes, custom_settings, _, _ = window._prepare_profile_writes()
            self.assertNotIn(LEGACY_LOW_RESOURCE_KEY, custom_settings)
            for key, value in chosen.items():
                self.assertIs(custom_settings[key], value, key)
            # And back: a saved visual profile loads into the switches.
            window._apply_profile_data("visual_effects", dict(VISUAL_PROFILE_DEFAULTS))
            self.assertFalse(any(window.inputs[key].isChecked() for key in SWITCHES))
            window._apply_profile_data("visual_effects", {**VISUAL_PROFILE_DEFAULTS, **chosen})
            self.assertEqual({key: window.inputs[key].isChecked() for key in SWITCHES}, chosen)
        finally:
            window._apply_profile_data("visual_effects", dict(VISUAL_PROFILE_DEFAULTS))

    def test_settings_files_and_profiles_with_the_old_switch_load_as_the_categories(self):
        gui = self.namespace["ConfigGUI"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "viewer_settings.json"
            cases = (({LEGACY_LOW_RESOURCE_KEY: True}, {key: True for key in OLD_CATEGORIES}),
                     ({LEGACY_LOW_RESOURCE_KEY: False}, {key: False for key in OLD_CATEGORIES}),
                     ({LEGACY_LOW_RESOURCE_KEY: True, "LOW_RESOURCE_WEBSERVER": False},
                      {"LOW_RESOURCE_WEBSERVER": False}))
            for saved, expected in cases:
                with self.subTest(saved=saved):
                    path.write_text(json.dumps({"NODE_SIZE": 12, **saved}), encoding="utf-8")
                    with mock.patch.dict(gui._read_custom_settings.__globals__,
                                         {"DEFAULT_SETTINGS_FILE": str(path)}):
                        loaded = gui._read_custom_settings(self.window)
                    self.assertEqual(loaded, {"NODE_SIZE": 12, **expected})
            # A saved Visual Effects profile.
            profile = Path(folder) / "old.json"
            profile.write_text(json.dumps({"NODE_SIZE": 12, LEGACY_LOW_RESOURCE_KEY: True}), encoding="utf-8")
            with mock.patch.object(self.window, "_profile_path", return_value=profile):
                data = self.window._named_profile_data("visual_effects", "old")
        self.assertEqual({key: data[key] for key in SWITCHES},
                         {"FEWER_EDGES_WHILE_MOVING": False, "SHOW_FPS": False,
                          **{key: True for key in OLD_CATEGORIES}})
        self.assertEqual(data["NODE_SIZE"], 12)

    def test_the_tips_say_what_each_category_covers(self):
        tips = self.window.tip_db_keys
        self.assertNotIn(LEGACY_LOW_RESOURCE_KEY, tips)
        text = {key: " ".join([tips[key].summary, *tips[key].details,
                               *(note.text for note in tips[key].notes)]) for key in SWITCHES}
        self.assertIn("`SSN_EDGE_MOTION_LOD=0`", text["FEWER_EDGES_WHILE_MOVING"])
        self.assertIn("Prints and captures", text["FEWER_EDGES_WHILE_MOVING"])
        self.assertIn("glide", text["LOW_RESOURCE_ANIMATIONS"])
        self.assertIn("slide", text["LOW_RESOURCE_ANIMATIONS"])
        for words in ("threshold slider", "neighbour colouring", "selection box"):
            self.assertIn(words, text["LOW_RESOURCE_DRAWING"])
        self.assertIn("Meta Data sheet", text["LOW_RESOURCE_WEBSERVER"])
        self.assertIn("Idle", text["SHOW_FPS"])
        # The old tip's promise, edges hidden while panning and zooming, is gone.
        for key in SWITCHES:
            self.assertNotIn("pan/zoom", text[key])
            self.assertNotIn("hides", text[key])

    def test_a_narrow_window_keeps_the_switches_whole(self):
        window = self.window
        size = window.size()
        try:
            window.resize(window.minimumSizeHint().width(), 900)
            self.app.processEvents()
            for row in self.rows():
                key = row.property("settingKey")
                switch, label = window.inputs[key], window.labels[key]
                with self.subTest(key=key):
                    self.assertGreaterEqual(label.width(), label.sizeHint().width())
                    self.assertLessEqual(switch.geometry().right(), row.width())
                    self.assertGreaterEqual(switch.width(), switch.sizeHint().width())
        finally:
            window.resize(size)
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
