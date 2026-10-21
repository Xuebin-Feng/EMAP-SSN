# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Config's Low Resource Mode section (EMAPSSN_Config.py, Visual Effects tab).

Under a padded line like the saved-config row's (the tab's sections have no
titles), a switch per category (Fewer Edges Drawn While Moving, No Zoom or
Sidebar Animation, Update Colors After Mouse Release, Update Web Pages After
Mouse Release) and Show Frame Rate (FPS), all off by default. The section has
its own two columns, apart from the colour grid above it: two switches to a
row in that order, each column's labels as wide as its longest and its
switches lined up with each other, not with the colour fields. These tests
check the section's place and spacing, its columns in English and Chinese,
that its labels leave the shared label column and the colour grid's second
column as they were, that it stacks by its own width, one switch a line in
order, saving and loading the switches, the old single switch in a settings
file or saved profile, the tips, and that no switch is greyed in UMAP mode.
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
from desktop.Viewer_Motion import MIN_SUBSET_SEGMENTS  # noqa: E402
from desktop.Viewer_State import (  # noqa: E402
    LEGACY_LOW_RESOURCE_KEY, LOW_RESOURCE_DEFAULTS, VISUAL_PROFILE_DEFAULTS, migrate_low_resource_mode,
)
from tests.theme_fixture import apply_theme_for_class  # noqa: E402

SWITCHES = ("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_ANIMATIONS", "LOW_RESOURCE_DRAWING",
            "LOW_RESOURCE_WEBSERVER", "SHOW_FPS")
OLD_CATEGORIES = tuple(LOW_RESOURCE_DEFAULTS)   # what LOW_RESOURCE_MODE stood for
LABELS = {"FEWER_EDGES_WHILE_MOVING": "Fewer Edges Drawn While Moving:",
          "LOW_RESOURCE_ANIMATIONS": "No Zoom or Sidebar Animation:",
          "LOW_RESOURCE_DRAWING": "Update Colors After Mouse Release:",
          "LOW_RESOURCE_WEBSERVER": "Update Web Pages After Mouse Release:",
          "SHOW_FPS": "Show Frame Rate (FPS):"}


class SectionGeometry:
    """The section's geometry, checked in a window the class opens (cls.window, cls.namespace).

    A mixin: the English and the Chinese test classes run these on windows
    built under each language.
    """

    # The switches' rows in the section, 2-2-1 in the user's order.
    ROWS = (("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_ANIMATIONS"),
            ("LOW_RESOURCE_DRAWING", "LOW_RESOURCE_WEBSERVER"), ("SHOW_FPS",))
    LEFT = ("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_DRAWING", "SHOW_FPS")
    RIGHT = ("LOW_RESOURCE_ANIMATIONS", "LOW_RESOURCE_WEBSERVER")
    COLOUR_SECOND_LABELS = ("INITIAL_NODE_COLOR", "CONNECTED_NODE_COLOR", "NODE_BOUNDARY_COLOR")
    # Whether the section's widest label is wider than the shared label column
    # (it is in English; Chinese labels are shorter than the column's 190 px).
    WIDER_THAN_LABEL_COLUMN = True

    def x(self, widget):
        from PySide6.QtCore import QPoint
        return widget.mapTo(self.window, QPoint(0, 0)).x()

    def y(self, widget):
        from PySide6.QtCore import QPoint
        return widget.mapTo(self.window, QPoint(0, 0)).y()

    def natural(self, label):
        """A label's own width: a fresh copy, as a fixed-width one hints its fixed width."""
        from PySide6.QtWidgets import QLabel, QStackedWidget
        if isinstance(label, QStackedWidget):       # the Inputs tab's per-mode labels
            return max(self.natural(label.widget(i)) for i in range(label.count()))
        fresh = QLabel(label.text())
        fresh.setFont(label.font())
        try:
            return fresh.sizeHint().width()
        finally:
            fresh.deleteLater()

    def field(self, key, window=None):
        """The cell a setting's input fills: its wrapper, or the input itself."""
        widget = (window or self.window).inputs[key]
        parent = widget.parentWidget()
        return parent if parent.objectName() == "wrapper" else widget

    def row_group(self, key, window=None):
        """The row layout's widget a switch's label is in."""
        return (window or self.window).labels[key].parentWidget()

    def assert_label_texts(self):
        raise NotImplementedError

    def long_label_window(self, size=None):
        """A window built with every label much longer, as a translation may make it, at
        the narrowest width the window allows: the section stacks there in any language."""
        QLabel = self.namespace["QLabel"]

        class LongLabel(QLabel):
            def __init__(self, text="", *args, **kwargs):
                if isinstance(text, str) and text:
                    text = f"{text} {'~' * max(len(text) // 2, 24)}"
                super().__init__(text, *args, **kwargs)

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        from tests.config_gui_loader import open_config_window
        with mock.patch.dict(self.namespace, {"QLabel": LongLabel}):
            window = open_config_window(self.namespace["ConfigGUI"], temp.name)
        window._cache_hash_request_id += 1
        for worker in window._cache_hash_workers.values():
            worker.requestInterruption()
            worker.wait()
        window._cache_hash_workers.clear()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        window.show()
        window.tabs.setCurrentIndex(1)
        window.resize(window.minimumSizeHint().width(), 1500)
        for _ in range(4):
            self.app.processEvents()
        return window

    def test_two_switches_a_row_in_the_sections_own_columns(self):
        window = self.window
        spacing = self.namespace["CONFIG_FIELD_HORIZONTAL_SPACING"]
        column_gap = self.namespace["LOW_RESOURCE_COLUMN_SPACING"]
        self.assert_label_texts()
        margin = self.x(window.labels["NODE_SIZE"])
        tops = [self.y(window.labels[keys[0]]) for keys in self.ROWS]
        self.assertEqual(tops, sorted(set(tops)))                 # the user's order, top to bottom
        for keys, top in zip(self.ROWS, tops):
            for key in keys:
                self.assertEqual(self.y(window.labels[key]), top, key)
        # 2-2-1: two switches in each of the first rows, Show Frame Rate alone in the last.
        rows_seen = [[key for key in SWITCHES if self.y(window.labels[key]) == top] for top in tops]
        self.assertEqual(rows_seen, [list(keys) for keys in self.ROWS])

        # The left column: labels at the tab's left margin, each as wide as the widest of
        # the column, and the switches in one line a spacing after them.
        left_width = max(self.natural(window.labels[key]) for key in self.LEFT)
        right_width = max(self.natural(window.labels[key]) for key in self.RIGHT)
        for key in self.LEFT:
            with self.subTest(key=key):
                self.assertEqual(self.x(window.labels[key]), margin)
                self.assertEqual(window.labels[key].width(), left_width)
                self.assertEqual(self.x(self.field(key)), margin + left_width + spacing)
        switch_field = self.field(self.LEFT[0])
        right_label_x = margin + left_width + spacing + switch_field.width() + column_gap
        # The columns are set well apart: as far as the colour rows' two columns are.
        self.assertGreaterEqual(column_gap, 24)
        # The rows are as far apart as the tab's other rows (the saved-config row's and the grid's).
        row_gap = self.namespace["CONFIG_TAB_ROW_SPACING"]
        first, second = (self.row_group(keys[0]) for keys in self.ROWS[:2])
        self.assertEqual(second.y() - (first.y() + first.height()), row_gap)
        for key in self.RIGHT:
            with self.subTest(key=key):
                self.assertEqual(self.x(window.labels[key]), right_label_x)
                self.assertEqual(window.labels[key].width(), right_width)
                self.assertEqual(self.x(self.field(key)), right_label_x + right_width + spacing)
        self.assertEqual({self.x(self.field(key)) for key in self.LEFT}, {margin + left_width + spacing})
        self.assertEqual(len({self.x(self.field(key)) for key in self.RIGHT}), 1)
        # Each switch sits at the start of its field, a field as wide as the switch.
        for key in SWITCHES:
            with self.subTest(key=key):
                switch = window.inputs[key]
                self.assertEqual(self.x(switch), self.x(self.field(key)))
                self.assertEqual(self.field(key).width(), switch.width())
        # Both columns stay in the tab's width, which no tab's content leaves.
        page = window.tabs.widget(1).widget()
        content_right = self.x(page) + page.width() - margin
        for key in SWITCHES:
            self.assertLessEqual(self.x(self.field(key)) + self.field(key).width(), content_right, key)

    def test_the_section_leaves_the_shared_columns_as_they_were(self):
        window = self.window
        section = {window.labels[key] for key in SWITCHES}
        column = list(dict.fromkeys(window._label_column))
        # Not in the window's shared label column, which is the longest of the other labels.
        self.assertFalse(section & set(column))
        self.assertEqual(window.label_column_width, max(self.natural(label) for label in column))
        # The colour grid's second column fits the colour labels alone ("Connected Node
        # Color:" in English), not "Update Web Pages After Mouse Release:".
        second = [window.labels[key] for key in self.COLOUR_SECOND_LABELS]
        widest = max(self.natural(label) for label in second)
        for label in second:
            with self.subTest(label=label.text()):
                self.assertEqual(label.width(), widest)
        section_widest = max(self.natural(label) for label in section)
        self.assertGreater(section_widest, widest)
        if self.WIDER_THAN_LABEL_COLUMN:
            self.assertGreater(section_widest, window.label_column_width)
        else:
            self.assertLess(section_widest, window.label_column_width)
        # The section is its own rows, below the colour rows' grid, not cells of it.
        for key in SWITCHES:
            self.assertTrue(self.row_group(key).objectName().startswith("lowResourceRow"), key)
            self.assertEqual(self.row_group(key).parentWidget().objectName(), "lowResourceSection")
        self.assertFalse(self.row_group("INITIAL_NODE_COLOR").objectName().startswith("lowResource"))

    def test_the_section_stacks_by_its_own_width_and_never_before_the_colour_rows(self):
        window = self.window
        size = window.size()
        colour = self.row_group("INITIAL_NODE_COLOR")
        rows = {key: self.row_group(key) for key in ("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_DRAWING")}
        # It needs only its own labels and switches, less than the colour rows' fields.
        for row in rows.values():
            self.assertLess(row.layout()._wide_width(), colour.layout()._wide_width())
        try:
            for width in range(window.minimumSizeHint().width(), 1700, 30):
                window.resize(width, 900)
                self.app.processEvents()
                for key, row in rows.items():
                    with self.subTest(width=width, key=key):
                        self.assertEqual(bool(row.property("stacked")),
                                         row.width() < row.layout()._wide_width())
                        if row.property("stacked"):
                            self.assertTrue(colour.property("stacked"))
                with self.subTest(width=width):
                    self.assertEqual(bool(colour.property("stacked")),
                                     colour.width() < colour.layout()._wide_width())
        finally:
            window.resize(size)
            self.app.processEvents()

    def test_a_narrow_window_stacks_the_switches_one_a_line_in_order(self):
        window = self.long_label_window()
        margin = self.x(window.labels["NODE_SIZE"])
        spacing = self.namespace["CONFIG_FIELD_HORIZONTAL_SPACING"]
        for key in ("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_DRAWING"):
            self.assertTrue(self.row_group(key, window).property("stacked"), key)
        tops = [self.y(window.labels[key]) for key in SWITCHES]
        self.assertEqual(tops, sorted(set(tops)))                    # one a line, in order 1 to 5
        longest = max(self.natural(window.labels[key]) for key in SWITCHES)
        for key in SWITCHES:
            switch, label = window.inputs[key], window.labels[key]
            with self.subTest(key=key):
                self.assertEqual(self.x(label), margin)
                self.assertGreaterEqual(label.width(), self.natural(label))
                self.assertGreaterEqual(switch.width(), switch.sizeHint().width())
                # Every label takes the longest one's room, so the switches line up.
                self.assertEqual(self.x(self.field(key, window)), margin + longest + spacing)
                self.assertEqual(self.x(switch), self.x(self.field(key, window)))

    def test_no_tab_needs_more_than_the_minimum_width_it_keeps(self):
        window = self.window
        floor = window.minimumSizeHint().width()
        for index in range(window.tabs.count()):
            scroll = window.tabs.widget(index)
            with self.subTest(tab=index):
                self.assertLessEqual(scroll.widget().minimumSizeHint().width(), scroll.minimumWidth())
        # The section does not set the window's minimum: without it the window needs the same width.
        section = window.tabs.widget(1).widget().findChild(QWidget, "lowResourceSection")
        try:
            section.setVisible(False)
            self.app.processEvents()
            self.assertEqual(window.minimumSizeHint().width(), floor)
        finally:
            section.setVisible(True)
            self.app.processEvents()


class ConfigSectionTests(SectionGeometry, unittest.TestCase):
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

    def assert_label_texts(self):
        for key, text in LABELS.items():
            self.assertEqual(self.window.labels[key].text(), text, key)

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
        # Under the last colour row, over the switches, with no title: the
        # tab's other sections have none.
        self.assertLess(self.y(self.window.inputs["NODE_BOUNDARY_COLOR"]), self.y(separator))
        self.assertLess(self.y(separator), self.y(self.window.inputs["FEWER_EDGES_WHILE_MOVING"]))
        self.assertIsNone(self.window.findChild(QLabel, "lowResourceTitle"))
        page = self.window.tabs.widget(1).widget()
        self.assertFalse([label.text() for label in page.findChildren(QLabel)
                          if label.text() in ("Low Resource Mode", "Low Resource Mode:")])

        # The rows above and below keep the saved-config line's distances to it.
        def gaps(wrapper):
            layout, index = self.containing_layout(wrapper.parentWidget().layout(), wrapper)
            above, below = layout.itemAt(index - 1).geometry(), layout.itemAt(index + 1).geometry()
            line = wrapper.geometry()
            return line.top() - above.bottom(), below.top() - line.bottom()
        self.assertEqual(gaps(separator.parentWidget()), gaps(saved_config.parentWidget()))

    def test_the_switches_are_toggle_switches_with_tips(self):
        from desktop.Desktop_App import ToggleSwitch
        for key in SWITCHES:
            with self.subTest(key=key):
                self.assertIsInstance(self.window.inputs[key], ToggleSwitch)
                self.assertIn(key, self.window.tip_db_keys)

    def test_the_section_is_clear_of_the_shared_label_column_in_the_wide_window_too(self):
        # In English the section's labels are wider than the shared label column, yet the
        # fields of every other row still start at that column.
        window = self.window
        self.assertGreater(max(self.natural(window.labels[key]) for key in SWITCHES),
                           window.label_column_width)
        self.assertEqual(self.x(self.field("NODE_SIZE")),
                         self.x(window.labels["NODE_SIZE"]) + window.label_column_width
                         + self.namespace["CONFIG_FIELD_HORIZONTAL_SPACING"])

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
        self.assertIn("node drag", text["LOW_RESOURCE_WEBSERVER"])
        self.assertIn("Idle", text["SHOW_FPS"])
        # The old tip's promise, edges hidden while panning and zooming, is gone.
        for key in SWITCHES:
            self.assertNotIn("pan/zoom", text[key])
            self.assertNotIn("hides", text[key])

    # What each switch still does in a Viewer opened in UMAP mode (Plot Manifold Instead
    # on) is shown in test_low_resource_mode.UmapModeTests: Fewer Edges Drawn While
    # Moving, for a selection of 250,000 edges or more; the zoom and sidebar
    # animations and the frame rate, in full; Update Colors After Mouse Release, for
    # the threshold slider and the selection box; Update Web Pages After Mouse
    # Release, for the selection box. Nodes cannot be dragged there. So no switch
    # has lost every effect, and none is greyed.

    def test_the_tips_tell_which_parts_do_not_apply_in_umap_mode(self):
        window = self.window
        umap = window.labels["UMAP_MODE"].text().rstrip(":")
        self.assertEqual(umap, "Plot Manifold Instead")
        tips = window.tip_db_keys

        def text(key):
            return " ".join([tips[key].summary, *tips[key].details, *(note.text for note in tips[key].notes)])

        for key in ("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_DRAWING", "LOW_RESOURCE_WEBSERVER"):
            with self.subTest(key=key):
                self.assertIn(umap, text(key))
                self.assertIn("cannot be dragged", text(key))
        # What still applies is named.
        self.assertIn("threshold slider and the selection box apply", text("LOW_RESOURCE_DRAWING"))
        self.assertIn("only the selection box applies", text("LOW_RESOURCE_WEBSERVER"))
        # Thinning starts at the Viewer's minimum, and a UMAP selection can reach it.
        self.assertIn(f"{MIN_SUBSET_SEGMENTS:,}", text("FEWER_EDGES_WHILE_MOVING"))
        self.assertIn(f"Fewer than {MIN_SUBSET_SEGMENTS:,} edges are always drawn in full.",
                      tips["FEWER_EDGES_WHILE_MOVING"].details)
        self.assertIn("selected nodes' edges", text("FEWER_EDGES_WHILE_MOVING"))
        # Switches with nothing to qualify say nothing of UMAP.
        for key in ("LOW_RESOURCE_ANIMATIONS", "SHOW_FPS"):
            self.assertNotIn("UMAP", text(key))

    def test_no_switch_is_greyed_in_umap_mode_and_none_changes_value(self):
        window = self.window
        chosen = {"FEWER_EDGES_WHILE_MOVING": True, "LOW_RESOURCE_ANIMATIONS": False,
                  "LOW_RESOURCE_DRAWING": True, "LOW_RESOURCE_WEBSERVER": True, "SHOW_FPS": False}
        for key, value in chosen.items():
            window.inputs[key].setChecked(value)

        def enabled():
            return {key: (window.labels[key].isEnabled(), window.inputs[key].isEnabled(),
                          self.field(key).isEnabled()) for key in SWITCHES}

        everything_on = {key: (True, True, True) for key in SWITCHES}
        inputs_defaults = dict(self.namespace["TAB_PROFILE_SPECS"]["inputs_outputs"]["defaults"])
        try:
            for shown_tab in (1, 0):                      # whichever tab is shown
                window.tabs.setCurrentIndex(shown_tab)
                self.app.processEvents()
                window.check_umap.setChecked(False)
                self.assertEqual(enabled(), everything_on)
                window.check_umap.setChecked(True)        # toggled
                self.assertEqual(enabled(), everything_on)
                window.check_umap.setChecked(False)
                self.assertEqual(enabled(), everything_on)
            # A saved Inputs profile loaded with UMAP on, then off.
            window._apply_profile_data("inputs_outputs", {**inputs_defaults, "UMAP_MODE": True})
            self.assertTrue(window.check_umap.isChecked())
            self.assertEqual(enabled(), everything_on)
            window._apply_profile_data("inputs_outputs", {**inputs_defaults, "UMAP_MODE": False})
            self.assertEqual(enabled(), everything_on)
            # Their values were never touched.
            self.assertEqual({key: window.inputs[key].isChecked() for key in SWITCHES}, chosen)
        finally:
            window.check_umap.setChecked(False)
            window.tabs.setCurrentIndex(1)
            window._apply_profile_data("visual_effects", dict(VISUAL_PROFILE_DEFAULTS))

    def test_a_window_opened_with_umap_on_shows_every_switch_enabled_on_either_tab(self):
        from tests.config_gui_loader import directory_globals, directory_settings
        gui = self.namespace["ConfigGUI"]
        globals_of_gui = gui._read_custom_settings.__globals__
        with tempfile.TemporaryDirectory() as folder:
            saved = {**directory_settings(folder), "UMAP_MODE": True, "UMAP_NEIGHBORS": 20,
                     "FEWER_EDGES_WHILE_MOVING": True, "LOW_RESOURCE_WEBSERVER": True}
            # open_config_window with these settings in place of the directory ones alone.
            with mock.patch.dict(globals_of_gui, directory_globals(globals_of_gui, folder)),                     mock.patch.object(gui, "_read_custom_settings", return_value=saved):
                window = gui()
            try:
                window._cache_hash_request_id += 1
                for worker in window._cache_hash_workers.values():
                    worker.requestInterruption()
                    worker.wait()
                window._cache_hash_workers.clear()
                window.show()
                for shown_tab in (0, 1, 2, 3):
                    window.tabs.setCurrentIndex(shown_tab)
                    self.app.processEvents()
                    self.assertTrue(window.check_umap.isChecked())
                    for key in SWITCHES:
                        with self.subTest(tab=shown_tab, key=key):
                            self.assertTrue(window.labels[key].isEnabled())
                            self.assertTrue(window.inputs[key].isEnabled())
                # The saved values show, and the UMAP-only fields are enabled with them.
                self.assertTrue(window.inputs["FEWER_EDGES_WHILE_MOVING"].isChecked())
                self.assertTrue(window.inputs["LOW_RESOURCE_WEBSERVER"].isChecked())
                self.assertFalse(window.inputs["LOW_RESOURCE_DRAWING"].isChecked())
                self.assertTrue(window.inputs["UMAP_NEIGHBORS"].isEnabled())
            finally:
                window.close()
                window.deleteLater()
                self.app.processEvents()


class ChineseConfigSectionTests(SectionGeometry, unittest.TestCase):
    """The section in a Config opened in Simplified Chinese (the catalog's drafts)."""

    # Its labels are shorter than the shared label column, 190 px in Chinese.
    WIDER_THAN_LABEL_COLUMN = False

    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Acceleration  # noqa: F401 - torch before PySide6
        from desktop.Desktop_App import install_translations
        from tests.config_gui_loader import load_config_namespace, open_config_window

        cls.namespace = load_config_namespace()
        cls.app = QApplication.instance() or QApplication([])
        font = cls.app.font()
        cls.addClassCleanup(cls.app.setFont, font)
        installed = install_translations(cls.app, "zh_CN")
        cls.addClassCleanup(installed.remove)
        cls.namespace["configure_qt_application_fonts"](cls.app)
        apply_theme_for_class(cls, cls.app)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.window = open_config_window(cls.namespace["ConfigGUI"], cls.temp.name)
        cls.addClassCleanup(cls.app.processEvents)
        cls.addClassCleanup(cls.window.close)
        cls.window._cache_hash_request_id += 1
        for worker in cls.window._cache_hash_workers.values():
            worker.requestInterruption()
            worker.wait()
        cls.window._cache_hash_workers.clear()
        cls.window.resize(1300, 900)
        cls.window.show()
        cls.window.tabs.setCurrentIndex(1)
        cls.app.processEvents()

    def assert_label_texts(self):
        for key, text in LABELS.items():
            shown = self.window.labels[key].text()
            self.assertNotEqual(shown, text, key)              # translated
            self.assertTrue(any("一" <= character <= "鿿" for character in shown), key)
            self.assertTrue(shown.endswith(("：", ":")), key)


if __name__ == "__main__":
    unittest.main()
