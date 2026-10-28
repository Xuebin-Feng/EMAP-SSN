# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Config's Visual Effects tab: its row order, and Edge Opacity's Auto switch.

Text Size is the first row, above Node Size. Beside Edge Opacity, at the start of
its cell as Step Size's Auto switch starts that cell, an Auto switch (setting
AUTO_EDGE_ALPHA, off by default) makes each edge's opacity follow its score. While
it is on the Edge Opacity slider and its box are greyed out and keep their value.
It is for the SSN layout method only: the Layout Method dropdown greys it out
for UMAP and PaCMAP, where it keeps its state and the slider is in use.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from desktop.Viewer_Edges import SCORE_ALPHA_MAX, SCORE_ALPHA_MIN  # noqa: E402
from tests.config_gui_loader import (  # noqa: E402
    choose_layout, directory_globals, directory_settings, load_config_namespace, open_config_window,
)
from tests.theme_fixture import apply_theme_for_class  # noqa: E402

SLIDER_KEYS = ("TEXT_SIZE", "NODE_SIZE", "EDGE_WIDTH", "NODE_BOUNDARY_WIDTH", "EDGE_ALPHA")


class EdgeOpacityConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_config_namespace()
        cls.gui_class = cls.namespace["ConfigGUI"]
        cls.app = cls.namespace["QApplication"].instance() or cls.namespace["QApplication"]([])
        cls.namespace["configure_qt_application_fonts"](cls.app)
        apply_theme_for_class(cls, cls.app)

    def open(self, saved=None):
        """A Config window on the Visual Effects tab; saved is what its custom profile reads."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        if saved is None:
            window = open_config_window(self.gui_class, folder.name)
        else:
            namespace = self.gui_class._read_custom_settings.__globals__
            with mock.patch.dict(namespace, directory_globals(namespace, folder.name)), mock.patch.object(
                self.gui_class, "_read_custom_settings", return_value={**directory_settings(folder.name), **saved}
            ):
                window = self.gui_class()
        window._cache_hash_request_id += 1
        for worker in window._cache_hash_workers.values():
            worker.requestInterruption()
            worker.wait()
        window._cache_hash_workers.clear()
        window.resize(1300, 900)
        window.show()
        window.tabs.setCurrentIndex(1)
        self.flush()
        self.addCleanup(self.flush)
        self.addCleanup(window.close)
        return window

    def flush(self):
        for _ in range(3):
            self.app.processEvents()

    def cell(self, window):
        return window.inputs["AUTO_EDGE_ALPHA"].parentWidget()

    def controls(self, window):
        """The Edge Opacity slider and its box."""
        slider = self.cell(window).layout().itemAt(2).widget()
        self.assertIs(self.cell(window).layout().itemAt(3).widget(), window.inputs["EDGE_ALPHA"])
        return slider, window.inputs["EDGE_ALPHA"]

    def state(self, window):
        """(Auto's switch enabled, its checked state, the slider enabled, the box enabled)."""
        slider, box = self.controls(window)
        switch = window.inputs["AUTO_EDGE_ALPHA"]
        return switch.isEnabled(), switch.isChecked(), slider.isEnabled(), box.isEnabled()

    # --- the rows ---------------------------------------------------------

    def test_text_size_is_the_first_row_above_node_size(self):
        window = self.open()
        page = window.tabs.currentWidget().widget()
        y = {key: window.labels[key].mapTo(page, window.labels[key].rect().topLeft()).y()
             for key in (*SLIDER_KEYS, "TEXT_COLOR", "EDGE_COLOR")}
        self.assertEqual(sorted(SLIDER_KEYS, key=y.get), list(SLIDER_KEYS))
        self.assertLess(y["EDGE_ALPHA"], y["TEXT_COLOR"])
        self.assertEqual(y["TEXT_SIZE"], min(y.values()))
        # One row apiece, evenly, with Text Size first in the section.
        steps = {y[after] - y[before] for before, after in zip(SLIDER_KEYS, SLIDER_KEYS[1:])}
        self.assertEqual(len(steps), 1)
        self.assertEqual(window.labels["TEXT_SIZE"].text(), "Text Size:")
        # Each slider's box is in its label's row.
        for key in SLIDER_KEYS:
            box, label = window.inputs[key], window.labels[key]
            self.assertAlmostEqual(box.mapTo(page, box.rect().center()).y(),
                                   label.mapTo(page, label.rect().center()).y(), delta=4)

    def test_the_catalog_lists_text_size_before_node_size(self):
        import xml.etree.ElementTree as ElementTree
        catalog = Path(__file__).resolve().parents[1] / "src" / "resources" / "languages" / "emapssn.ts"
        sources = [message.findtext("source") for message in ElementTree.parse(catalog).getroot().iter("message")]
        self.assertLess(sources.index("Text Size:"), sources.index("Node Size:"))

    # --- the switch -------------------------------------------------------

    def test_the_switch_starts_the_edge_opacity_cell_as_the_step_size_switch_starts_its(self):
        window = self.open()
        switch = window.inputs["AUTO_EDGE_ALPHA"]
        step_switch = window.inputs["AUTO_DT"]
        ToggleSwitch = self.namespace["ToggleSwitch"]
        self.assertIsInstance(switch, ToggleSwitch)
        self.assertIs(window.inputs["AUTO_DT"].__class__, switch.__class__)
        self.assertEqual((switch.width(), switch.height()), (step_switch.width(), step_switch.height()))
        self.assertEqual(switch.text(), "Auto OFF")
        self.assertEqual(switch.state_texts(), ("Auto OFF", "Auto ON"))
        self.assertFalse(switch.isChecked())
        cell = self.cell(window)
        self.assertEqual(cell.objectName(), "wrapper")
        slider, box = self.controls(window)
        self.assertIs(cell.layout().itemAt(0).widget(), switch)
        # A field spacing between the switch and the slider.
        self.assertEqual(slider.x() - (switch.x() + switch.width()),
                         self.namespace["CONFIG_FIELD_HORIZONTAL_SPACING"])
        # It is the Edge Opacity row's: the slider, box and label are.
        self.assertIs(window.labels["EDGE_ALPHA"].parentWidget(), cell.parentWidget())
        # The boxes of the slider rows stay in line, as do the sliders' ends.
        boxes = {key: window.inputs[key].mapTo(window, window.inputs[key].rect().topLeft()).x()
                 for key in SLIDER_KEYS}
        self.assertEqual(len(set(boxes.values())), 1, boxes)
        # The switch is where the other rows' sliders start.
        other = window.inputs["NODE_SIZE"].parentWidget()
        self.assertEqual(switch.mapTo(window, switch.rect().topLeft()).x(),
                         other.layout().itemAt(0).widget().mapTo(window, switch.rect().topLeft()).x())

    def test_auto_greys_the_slider_and_its_box_and_keeps_their_value(self):
        window = self.open()
        slider, box = self.controls(window)
        switch = window.inputs["AUTO_EDGE_ALPHA"]
        box.setValue(0.37)
        self.assertEqual(slider.value(), 37)
        self.assertEqual(self.state(window), (True, False, True, True))
        switch.setChecked(True)
        self.flush()
        self.assertEqual(switch.text(), "Auto ON")
        self.assertEqual(self.state(window), (True, True, False, False))
        # The label stays as it was, as Step Size's does; the value is kept and saved.
        self.assertTrue(window.labels["EDGE_ALPHA"].isEnabled())
        self.assertEqual((box.value(), slider.value()), (0.37, 37))
        saved = window.collect_data()
        self.assertIs(saved["AUTO_EDGE_ALPHA"], True)
        self.assertEqual(saved["EDGE_ALPHA"], "0.37")
        self.assertEqual(window._collect_tab_profile_data("visual_effects")["EDGE_ALPHA"], "0.37")
        switch.setChecked(False)
        self.flush()
        self.assertEqual(self.state(window), (True, False, True, True))
        self.assertEqual((box.value(), slider.value()), (0.37, 37))
        self.assertIs(window.collect_data()["AUTO_EDGE_ALPHA"], False)

    def test_the_saved_state_shows_at_start_and_is_what_a_save_writes(self):
        for saved in (True, False):
            with self.subTest(saved=saved):
                window = self.open({"AUTO_EDGE_ALPHA": saved, "EDGE_ALPHA": "0.6"})
                self.assertEqual(self.state(window), (True, saved, not saved, not saved))
                self.assertEqual(window.inputs["EDGE_ALPHA"].value(), 0.6)
                self.assertIs(window._collect_tab_profile_data("visual_effects")["AUTO_EDGE_ALPHA"], saved)
                self.assertIs(window.collect_data()["AUTO_EDGE_ALPHA"], saved)
        # Saved as text, as a settings file written by hand may have it.
        window = self.open({"AUTO_EDGE_ALPHA": "true"})
        self.assertEqual(self.state(window), (True, True, False, False))

    def test_the_module_reads_the_saved_setting_as_a_flag(self):
        names = self.namespace
        self.assertIs(names["AUTO_EDGE_ALPHA"], False)
        previous = names["AUTO_EDGE_ALPHA"]
        self.addCleanup(names.__setitem__, "AUTO_EDGE_ALPHA", previous)
        for value, expected in (("true", True), (True, True), ("false", False), ("0", False)):
            with self.subTest(value=value):
                names["apply_viewer_settings"]({"AUTO_EDGE_ALPHA": value})
                self.assertIs(names["AUTO_EDGE_ALPHA"], expected)

    def test_profiles_hold_it_and_older_ones_load_as_off(self):
        window = self.open()
        spec = self.namespace["TAB_PROFILE_SPECS"]["visual_effects"]["defaults"]
        self.assertIs(spec["AUTO_EDGE_ALPHA"], False)
        # A profile saved before the switch existed.
        older = {key: value for key, value in spec.items() if key != "AUTO_EDGE_ALPHA"}
        loaded = window._normalize_profile_data("visual_effects", older)
        self.assertIs(loaded["AUTO_EDGE_ALPHA"], False)
        for text, expected in (("true", True), ("yes", True), ("false", False), (True, True)):
            self.assertIs(window._normalize_profile_data(
                "visual_effects", {"AUTO_EDGE_ALPHA": text})["AUTO_EDGE_ALPHA"], expected)
        with self.assertRaisesRegex(ValueError, "invalid value for AUTO_EDGE_ALPHA: expected true or false"):
            window._normalize_profile_data("visual_effects", {"AUTO_EDGE_ALPHA": "maybe"})
        # Applying a profile sets the switch and, through it, the slider.
        window._apply_profile_data("visual_effects", {**spec, "AUTO_EDGE_ALPHA": True, "EDGE_ALPHA": 0.4})
        self.assertEqual(self.state(window), (True, True, False, False))
        self.assertEqual(window.inputs["EDGE_ALPHA"].value(), 0.4)
        window._apply_profile_data("visual_effects", dict(spec))
        self.assertEqual(self.state(window), (True, False, True, True))

    def test_the_built_in_default_profile_turns_it_off_and_locks_the_tab(self):
        window = self.open({"AUTO_EDGE_ALPHA": True})
        self.assertEqual(self.state(window), (True, True, False, False))
        selector = window.profile_selectors["visual_effects"]
        self.addCleanup(selector.setCurrentText, "(custom)")
        selector.setCurrentText("(default)")
        self.flush()
        switch = window.inputs["AUTO_EDGE_ALPHA"]
        self.assertFalse(switch.isChecked())
        self.assertFalse(switch.isEnabled())  # the profile is read-only
        self.assertEqual(switch.styleSheet(), window.inputs["AUTO_DT"].styleSheet())
        selector.setCurrentText("(new)")
        self.flush()
        self.assertEqual(self.state(window), (True, False, True, True))
        switch.setChecked(True)
        self.assertEqual(self.state(window), (True, True, False, False))

    # --- the layout method ------------------------------------------------

    def test_the_switch_is_for_the_ssn_layout_method_only(self):
        for auto in (False, True):
            with self.subTest(auto=auto):
                window = self.open({"AUTO_EDGE_ALPHA": auto})
                self.assertFalse(window.uses_manifold_layout())
                # SSN: the switch works, and the slider is greyed out while Auto is on.
                self.assertEqual(self.state(window), (True, auto, not auto, not auto))
                for method in ("umap", "pacmap"):
                    choose_layout(window, method)
                    self.flush()
                    # Greyed out, still showing and saving its state; the slider is in use.
                    self.assertEqual(self.state(window), (False, auto, True, True), method)
                    self.assertIs(window.collect_data()["AUTO_EDGE_ALPHA"], auto)
                    self.assertIs(window._collect_tab_profile_data("visual_effects")["AUTO_EDGE_ALPHA"], auto)
                    choose_layout(window, "ssn")
                    self.flush()
                    # Back to SSN, the state is back too.
                    self.assertEqual(self.state(window), (True, auto, not auto, not auto), method)
                # From UMAP to PaCMAP and back, the switch stays greyed out.
                choose_layout(window, "umap")
                choose_layout(window, "pacmap")
                self.assertEqual(self.state(window), (False, auto, True, True))
                choose_layout(window, "ssn")
                self.assertEqual(self.state(window), (True, auto, not auto, not auto))

    def test_the_switch_is_not_for_a_manifold_layout_even_when_it_is_flipped_there(self):
        window = self.open()
        choose_layout(window, "umap")
        self.flush()
        # The switch cannot be clicked, but a profile may set it: still greyed out, slider in use.
        window.inputs["AUTO_EDGE_ALPHA"].setChecked(True)
        self.flush()
        self.assertEqual(self.state(window), (False, True, True, True))
        choose_layout(window, "ssn")
        self.assertEqual(self.state(window), (True, True, False, False))

    def test_a_window_opened_in_a_manifold_layout_has_it_greyed_out_with_the_saved_state(self):
        window = self.open({"UMAP_MODE": True, "UMAP_NEIGHBORS": 20, "AUTO_EDGE_ALPHA": True})
        self.assertTrue(window.uses_manifold_layout())
        self.assertEqual(self.state(window), (False, True, True, True))

    def test_a_saved_inputs_profile_with_umap_greys_it_out_as_the_dropdown_does(self):
        window = self.open({"AUTO_EDGE_ALPHA": True})
        inputs = dict(self.namespace["TAB_PROFILE_SPECS"]["inputs_outputs"]["defaults"])
        window._apply_profile_data("inputs_outputs", {**inputs, "UMAP_MODE": True})
        self.assertEqual(self.state(window), (False, True, True, True))
        window._apply_profile_data("inputs_outputs", {**inputs, "UMAP_MODE": False})
        self.assertEqual(self.state(window), (True, True, False, False))

    def test_the_layout_method_change_uses_the_path_that_greys_the_other_ssn_only_controls(self):
        window = self.open()
        calls = []
        original = window._update_edge_opacity_controls
        with mock.patch.object(window, "_update_edge_opacity_controls",
                               side_effect=lambda *args: (calls.append(args), original(*args))[1]):
            choose_layout(window, "umap")
        self.assertTrue(calls)
        # The same change greys the Simulation & Physics tab's SSN-only controls.
        self.assertFalse(window.inputs["LAYOUT_DEVICE_SELECTION"].isEnabled())

    # --- the tip ----------------------------------------------------------

    def test_the_tip_says_what_it_does_and_names_the_layout_method(self):
        window = self.open()
        tip = window.tip_db_keys["AUTO_EDGE_ALPHA"]
        low, high = round(SCORE_ALPHA_MIN * 100), round(SCORE_ALPHA_MAX * 100)
        self.assertEqual((low, high), (15, 85))
        self.assertIn(f"{low}% at the threshold", tip.summary)
        self.assertIn(f"{high}% at the top score", tip.summary)
        self.assertIn("Edge Opacity slider is then not used", tip.summary)
        details = " ".join(tip.details)
        self.assertIn("Only for the SSN layout method", details)
        self.assertIn("UMAP or PaCMAP", details)
        # It is the switch's tip, and the Edge Opacity cell's other fields keep theirs.
        self.assertIsNot(tip, window.tip_db_keys["EDGE_ALPHA"])
        self.assertIn("EDGE_ALPHA", window.tip_db_keys)

    # --- the geometry -----------------------------------------------------

    def resize_panel(self, window, width):
        window.resize(width + 360, 850)
        window.main_split.setSizes([width + 4, 326])
        self.flush()
        delta = width - window.tabs.currentWidget().width()
        left, right = window.main_split.sizes()
        window.main_split.setSizes([left + delta, right - delta])
        self.flush()

    def test_the_cell_fits_at_every_width_and_the_page_never_scrolls_sideways(self):
        window = self.open()
        self.resize_panel(window, 0)
        narrowest = window.tabs.currentWidget().width()
        self.assertLessEqual(narrowest, 600 + 4)
        for width in (600, 800, 1000, 1400, 600):
            with self.subTest(width=width):
                self.resize_panel(window, width)
                scroll = window.tabs.currentWidget()
                self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
                cell = self.cell(window)
                slider, box = self.controls(window)
                switch = window.inputs["AUTO_EDGE_ALPHA"]
                for child in (switch, slider, box):
                    self.assertTrue(cell.rect().contains(child.geometry()), child)
                self.assertFalse(switch.geometry().intersects(slider.geometry()))
                self.assertFalse(slider.geometry().intersects(box.geometry()))
                self.assertGreaterEqual(slider.width(), slider.minimumWidth())
                # Its row is as wide as the others, whatever the width.
                other = window.inputs["EDGE_WIDTH"].parentWidget()
                self.assertEqual((cell.x(), cell.width()), (other.x(), other.width()))


if __name__ == "__main__":
    unittest.main()
