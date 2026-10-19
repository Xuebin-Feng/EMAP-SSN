# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Config's Low Resource Mode section (EMAPSSN_Config.py, Visual Effects tab).

Under the same padded line as the saved-config row and a "Low Resource Mode"
heading, a switch per category (Fewer Edges While Moving, Basic Animations,
Node and Edge Drawing, Webserver Communication) and Show FPS, on the tab's
grid two to a row in the colour rows' columns, all off by default. These
tests check the section's place and spacing, the heading's alignment, the
rows and columns (no column wider than before), saving and loading the
switches, the old single switch in a settings file or saved profile, the
tips, and the stacked order in a narrow window.
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

    # The switches' rows on the Visual Effects grid: (1, 2), (3), (4, 5).
    ROWS = (("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_ANIMATIONS"), ("LOW_RESOURCE_DRAWING",),
            ("LOW_RESOURCE_WEBSERVER", "SHOW_FPS"))
    COLOUR_SECOND_LABELS = ("INITIAL_NODE_COLOR", "CONNECTED_NODE_COLOR", "NODE_BOUNDARY_COLOR")

    def x(self, widget):
        from PySide6.QtCore import QPoint
        return widget.mapTo(self.window, QPoint(0, 0)).x()

    def y(self, widget):
        from PySide6.QtCore import QPoint
        return widget.mapTo(self.window, QPoint(0, 0)).y()

    def natural(self, label):
        """A label's own width: a fresh copy, as a fixed-width one hints its fixed width."""
        from PySide6.QtWidgets import QLabel
        fresh = QLabel(label.text())
        fresh.setFont(label.font())
        try:
            return fresh.sizeHint().width()
        finally:
            fresh.deleteLater()

    def field(self, key):
        """The cell a setting's input fills: its wrapper, or the input itself."""
        widget = self.window.inputs[key]
        parent = widget.parentWidget()
        return parent if parent.objectName() == "wrapper" else widget

    @staticmethod
    def containing_layout(layout, widget):
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget() is widget:
                return layout, index
            if item.layout() is not None:
                found = ConfigSectionTests.containing_layout(item.layout(), widget)
                if found is not None:
                    return found
        return None

    def test_the_section_sits_under_a_line_like_the_saved_config_one(self):
        from PySide6.QtWidgets import QFrame, QLabel
        separator = self.window.findChild(QFrame, "lowResourceSeparator")
        saved_config = self.window.profile_separators["visual_effects"]
        self.assertIs(separator, self.window.low_resource_separator)
        self.assertEqual(separator.property("role"), "separator")
        self.assertEqual(separator.height(), saved_config.height())
        self.assertEqual(separator.parentWidget().layout().contentsMargins(),
                         saved_config.parentWidget().layout().contentsMargins())
        # Under the last colour row, over the heading, over the switches.
        title = self.window.findChild(QLabel, "lowResourceTitle")
        self.assertEqual(title.text(), "Low Resource Mode")
        self.assertEqual(title.property("role"), "title")
        self.assertLess(self.y(self.window.inputs["NODE_BOUNDARY_COLOR"]), self.y(separator))
        self.assertLess(self.y(separator), self.y(title))
        self.assertLess(self.y(title), self.y(self.window.inputs["FEWER_EDGES_WHILE_MOVING"]))

        # The rows above and below keep the saved-config line's distances to it.
        def gaps(wrapper):
            layout, index = self.containing_layout(wrapper.parentWidget().layout(), wrapper)
            above, below = layout.itemAt(index - 1).geometry(), layout.itemAt(index + 1).geometry()
            line = wrapper.geometry()
            return line.top() - above.bottom(), below.top() - line.bottom()
        self.assertEqual(gaps(separator.parentWidget()), gaps(saved_config.parentWidget()))

    def test_the_heading_starts_where_the_labels_do(self):
        from PySide6.QtWidgets import QLabel
        title = self.window.findChild(QLabel, "lowResourceTitle")
        # No indent and no left padding: the text starts at the widget's edge.
        self.assertEqual(title.indent(), 0)
        self.assertEqual(title.contentsRect().left(), 0)
        for key in ("NODE_SIZE", "FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_DRAWING"):
            self.assertEqual(self.x(title), self.x(self.window.labels[key]), key)
        self.assertEqual(self.x(title), self.x(self.window.profile_labels["visual_effects"]))
        # Still the Config's title style otherwise.
        self.assertTrue(title.font().bold() or title.font().weight() >= 600)

    def test_two_switches_a_row_in_the_colour_rows_columns(self):
        from desktop.Desktop_App import ToggleSwitch
        window = self.window
        labels = {"FEWER_EDGES_WHILE_MOVING": "Fewer Edges While Moving:",
                  "LOW_RESOURCE_ANIMATIONS": "Basic Animations:",
                  "LOW_RESOURCE_DRAWING": "Node and Edge Drawing:",
                  "LOW_RESOURCE_WEBSERVER": "Webserver Communication:", "SHOW_FPS": "Show FPS:"}
        rows = [self.y(window.labels[keys[0]]) for keys in self.ROWS]
        self.assertEqual(rows, sorted(rows))                     # the user's order, top to bottom
        first_x = self.x(self.field("NODE_SIZE"))
        second_label_x = self.x(window.labels["INITIAL_NODE_COLOR"])
        second_field_x = self.x(self.field("INITIAL_NODE_COLOR"))
        for keys in self.ROWS:
            for slot, key in enumerate(keys):
                with self.subTest(key=key):
                    switch, label = window.inputs[key], window.labels[key]
                    self.assertIsInstance(switch, ToggleSwitch)
                    self.assertEqual(label.text(), labels[key])
                    self.assertEqual(self.y(label), self.y(window.labels[keys[0]]))
                    self.assertIn(key, window.tip_db_keys)
                    if slot == 0:
                        self.assertEqual(label.width(), window.label_column_width)
                        self.assertEqual(self.x(self.field(key)), first_x)
                    else:
                        # Under "Initial Node Color:", its switch under the colour fields.
                        self.assertEqual(self.x(label), second_label_x)
                        self.assertEqual(self.x(self.field(key)), second_field_x)
        # Node and Edge Drawing has its row to itself.
        drawing_row = self.y(window.labels["LOW_RESOURCE_DRAWING"])
        self.assertEqual([key for key in SWITCHES if self.y(window.labels[key]) == drawing_row],
                         ["LOW_RESOURCE_DRAWING"])

    def test_no_column_gets_wider(self):
        window = self.window
        switch_labels = {window.labels[key] for key in SWITCHES}
        # The first column is still the longest of the other labels (in English,
        # "MSA Input (.fasta / _sparse.h5):", 174 px on Windows).
        others = [label for label in dict.fromkeys(window._label_column)
                  if label not in switch_labels and hasattr(label, "text")]
        self.assertEqual(window.label_column_width, max(self.natural(label) for label in others))
        self.assertTrue(all(self.natural(label) < window.label_column_width for label in switch_labels))
        # The second column is still the longest colour label's.
        colour_width = max(self.natural(window.labels[key]) for key in self.COLOUR_SECOND_LABELS)
        for key in self.COLOUR_SECOND_LABELS + ("LOW_RESOURCE_ANIMATIONS", "SHOW_FPS"):
            with self.subTest(key=key):
                self.assertEqual(window.labels[key].width(), colour_width)
        for key in ("LOW_RESOURCE_ANIMATIONS", "SHOW_FPS"):
            self.assertLess(self.natural(window.labels[key]), colour_width)

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
        self.assertIn("selection box", text["LOW_RESOURCE_WEBSERVER"])
        self.assertIn("Idle", text["SHOW_FPS"])
        # The old tip's promise, edges hidden while panning and zooming, is gone.
        for key in SWITCHES:
            self.assertNotIn("pan/zoom", text[key])
            self.assertNotIn("hides", text[key])

    def test_the_switch_rows_stack_when_the_colour_rows_do(self):
        window = self.window
        size = window.size()
        rows = {key: window.labels[key].parentWidget()
                for key in ("INITIAL_NODE_COLOR", "FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_WEBSERVER")}
        seen = set()
        try:
            for width in range(window.minimumSizeHint().width(), 1600, 40):
                window.resize(width, 900)
                self.app.processEvents()
                stacked = {key: bool(row.property("stacked")) for key, row in rows.items()}
                with self.subTest(width=width):
                    self.assertEqual(len(set(stacked.values())), 1, stacked)
                seen.add(stacked["INITIAL_NODE_COLOR"])
        finally:
            window.resize(size)
            self.app.processEvents()
        self.assertEqual(seen, {False, True})        # the sweep saw both layouts

    def test_a_narrow_window_stacks_the_switches_in_order(self):
        window = self.window
        size = window.size()
        try:
            window.resize(window.minimumSizeHint().width(), 900)
            self.app.processEvents()
            margin = self.x(window.labels["NODE_SIZE"])
            tops = [self.y(window.labels[key]) for key in SWITCHES]
            self.assertEqual(tops, sorted(tops))
            self.assertEqual(len(set(tops)), len(SWITCHES))      # one a line
            for key in SWITCHES:
                switch, label = window.inputs[key], window.labels[key]
                with self.subTest(key=key):
                    self.assertEqual(self.x(label), margin)
                    self.assertGreaterEqual(label.width(), self.natural(label))
                    self.assertGreaterEqual(switch.width(), switch.sizeHint().width())
                    self.assertEqual(self.x(self.field(key)), self.x(self.field("NODE_SIZE")))
        finally:
            window.resize(size)
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
