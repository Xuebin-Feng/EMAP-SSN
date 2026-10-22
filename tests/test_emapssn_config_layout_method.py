"""The Config's Layout Method dropdown: SSN, UMAP or PaCMAP.

On the Inputs tab the dropdown Layout Method ends the edge-filter row. SSN, the default, is the physics
layout; UMAP and PaCMAP are the manifold layouts. The saved settings stay two: UMAP_MODE ("a manifold layout
instead of physics") and LAYOUT_METHOD (which manifold method). The dropdown reads and writes both: SSN is
UMAP_MODE off and leaves LAYOUT_METHOD as it was saved, UMAP and PaCMAP are UMAP_MODE on with their own method.
Everything that followed the old Plot Manifold Instead switch follows the dropdown: the two edge-filter fields
(UMAP's neighbour count and minimum distance, or PaCMAP's neighbour count with the minimum distance greyed
out, in place of the threshold and Top Edge %), the Simulation & Physics tab, the cache identity the window
works out, saved settings, profiles and what a layout job is handed. Anything saved before the method existed
loads as UMAP.
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

from tests.config_gui_loader import (
    choose_layout, directory_globals, directory_settings, load_config_namespace, open_config_window,
)
from tests.theme_fixture import apply_theme_for_class

# What each choice means as the two saved settings: (UMAP_MODE, LAYOUT_METHOD).
SAVED_AS = {"ssn": (False, None), "umap": (True, "umap"), "pacmap": (True, "pacmap")}


class LayoutMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_config_namespace()
        cls.gui_class = cls.namespace["ConfigGUI"]
        cls.app = cls.namespace["QApplication"].instance() or cls.namespace["QApplication"]([])
        cls.namespace["configure_qt_application_fonts"](cls.app)
        apply_theme_for_class(cls, cls.app)

    def open(self, saved=None):
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
        self.app.processEvents()
        self.addCleanup(self.app.processEvents)
        self.addCleanup(window.close)
        return window

    def page(self, window, label_stack):
        """The text of the page a stacked label shows."""
        return label_stack.currentWidget().text()

    def row(self, window):
        """The stacked pairs of the filter row: (first label, first field, second label, second field)."""
        stacks = window._input_mode_stacks
        return stacks[0], stacks[1], stacks[2], stacks[3]

    def saved_settings(self, window):
        """The two settings the window would save: (UMAP_MODE, LAYOUT_METHOD), by every route."""
        profile = (window._widget_profile_value("UMAP_MODE"), window._widget_profile_value("LAYOUT_METHOD"))
        collected = window.collect_data()
        tab = window._collect_tab_profile_data("inputs_outputs")
        self.assertEqual(profile, (collected["UMAP_MODE"], collected["LAYOUT_METHOD"]))
        self.assertEqual(profile, (tab["UMAP_MODE"], tab["LAYOUT_METHOD"]))
        carried = window.language_carry_over()["values"]
        self.assertEqual(profile, (carried["UMAP_MODE"], carried["LAYOUT_METHOD"]))
        return profile

    # --- the widget ----------------------------------------------------------------------------

    def test_the_dropdown_and_its_defaults(self):
        window = self.open()
        combo = window.cb_layout_method
        self.assertEqual([combo.itemData(i) for i in range(combo.count())], ["ssn", "umap", "pacmap"])
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["SSN", "UMAP", "PaCMAP"])
        self.assertEqual(combo.currentData(), "ssn")
        self.assertFalse(window.uses_manifold_layout())
        self.assertEqual((window.spin_pacmap_k.minimum(), window.spin_pacmap_k.maximum(), window.spin_pacmap_k.value()),
                         (2, 50, 10))
        self.assertEqual(window.labels["LAYOUT_METHOD"].text(), "Layout Method:")
        self.assertEqual(window.labels["PACMAP_NEIGHBORS"].text(), "PaCMAP Nearest Neighbors:")
        # The old switch and the full-width row under the filters are gone.
        self.assertFalse(hasattr(window, "check_umap"))
        self.assertNotIn("UMAP_MODE", window.labels)
        self.assertNotIn("UMAP_MODE", window.inputs)
        self.assertIs(window.inputs["LAYOUT_METHOD"], combo)
        self.assertIs(combo.parentWidget(), window.findChild(self.namespace["QWidget"], "filterRow"))
        self.assertIs(window.labels["LAYOUT_METHOD"].parentWidget(), combo.parentWidget())

    def test_the_default_profile_is_ssn_and_keeps_umap_as_the_method_for_later(self):
        defaults = self.namespace["TAB_PROFILE_SPECS"]["inputs_outputs"]["defaults"]
        self.assertIs(defaults["UMAP_MODE"], False)
        self.assertEqual(defaults["LAYOUT_METHOD"], "umap")
        self.assertEqual(self.namespace["layout_choice"](defaults["UMAP_MODE"], defaults["LAYOUT_METHOD"]), "ssn")
        window = self.open()
        window._apply_profile_data("inputs_outputs", dict(defaults))
        self.assertEqual(window.cb_layout_method.currentData(), "ssn")
        self.assertEqual(self.saved_settings(window), (False, "umap"))

    def test_each_choice_shows_its_own_fields(self):
        window = self.open()
        first_label, first_field, second_label, second_field = self.row(window)
        self.assertEqual(first_label.count(), 3)
        self.assertEqual(first_field.count(), 3)
        self.assertEqual(second_label.count(), 2)
        self.assertEqual(second_field.count(), 2)

        def shown():
            return (self.page(window, first_label), self.page(window, second_label),
                    first_field.currentWidget() is window.spin_pacmap_k, first_field.currentWidget() is window.spin_umap_k)

        self.assertEqual(shown()[:2], ("Similarity Threshold:", "Top Edge %:"))
        choose_layout(window, "umap")
        self.assertEqual(shown(), ("UMAP Nearest Neighbors:", "UMAP Min Distance:", False, True))
        choose_layout(window, "pacmap")
        self.assertEqual(shown(), ("PaCMAP Nearest Neighbors:", "UMAP Min Distance:", True, False))
        choose_layout(window, "umap")
        self.assertEqual(shown()[:2], ("UMAP Nearest Neighbors:", "UMAP Min Distance:"))
        choose_layout(window, "pacmap")
        choose_layout(window, "ssn")
        # Physics ignores the method: the filters show, and the manifold fields are off.
        self.assertEqual(shown()[:2], ("Similarity Threshold:", "Top Edge %:"))

    def test_what_is_enabled_follows_the_dropdown_live(self):
        window = self.open()
        keys = ("UMAP_NEIGHBORS", "UMAP_MIN_DIST", "PACMAP_NEIGHBORS")

        def enabled():
            return {key: (window.inputs[key].isEnabled(), window.labels[key].isEnabled()) for key in keys}

        both = lambda value: (value, value)       # noqa: E731
        choose_layout(window, "ssn")
        self.assertEqual(enabled(), {key: both(False) for key in keys})
        self.assertTrue(window.spin_thresh.isEnabled())
        self.assertTrue(window.labels["SIMILARITY_THRESHOLD"].isEnabled())
        self.assertTrue(window.spin_top.isEnabled())
        self.assertTrue(window.labels["TOP_EDGE_PERCENT"].isEnabled())
        self.assertTrue(window.tabs.isTabVisible(2))
        self.assertTrue(window.cb_layout_device.isEnabled())
        choose_layout(window, "umap")
        self.assertEqual(enabled(), {"UMAP_NEIGHBORS": both(True), "UMAP_MIN_DIST": both(True),
                                     "PACMAP_NEIGHBORS": both(False)})
        self.assertFalse(window.spin_thresh.isEnabled())
        self.assertFalse(window.spin_top.isEnabled())
        self.assertFalse(window.tabs.isTabVisible(2))
        self.assertFalse(window.cb_layout_device.isEnabled())
        choose_layout(window, "pacmap")
        # PaCMAP has no minimum distance: its field and label stay, greyed out.
        self.assertEqual(enabled(), {"UMAP_NEIGHBORS": both(False), "UMAP_MIN_DIST": both(False),
                                     "PACMAP_NEIGHBORS": both(True)})
        self.assertTrue(window.inputs["UMAP_MIN_DIST"].isVisibleTo(window))
        # The physics tab is hidden for either method, as before.
        self.assertFalse(window.tabs.isTabVisible(2))
        # The dropdown itself is never greyed: it is how a layout is chosen.
        for choice in ("ssn", "umap", "pacmap"):
            choose_layout(window, choice)
            self.assertTrue(window.cb_layout_method.isEnabled())
            self.assertTrue(window.labels["LAYOUT_METHOD"].isEnabled())
        choose_layout(window, "ssn")
        self.assertTrue(window.tabs.isTabVisible(2))

    def test_the_physics_tab_is_left_when_a_manifold_method_is_chosen_on_it(self):
        window = self.open()
        window.tabs.setCurrentIndex(2)
        choose_layout(window, "pacmap")
        self.assertNotEqual(window.tabs.currentIndex(), 2)

    def test_the_cache_identity_follows_the_dropdown(self):
        window = self.open()
        self.assertEqual({key: window._cache_setting_values()[key]
                          for key in ("umap_mode", "layout_method")}, {"umap_mode": False, "layout_method": "umap"})
        choose_layout(window, "pacmap")
        window.spin_pacmap_k.setValue(23)
        values = window._cache_setting_values()
        self.assertEqual((values["layout_method"], values["pacmap_neighbors"], values["umap_mode"]), ("pacmap", 23, True))
        # The cache identity the window works out is the one a layout job writes.
        from Cache_Manifest import build_compatibility
        compatibility = build_compatibility("a" * 64, "b" * 64, "alignment", **values)
        self.assertEqual(compatibility["layout_mode"], "pacmap")
        self.assertEqual(compatibility["edge_filter"], {"mode": "pacmap_neighbors", "value": 23})
        window.spin_pacmap_k.setValue(24)
        self.assertEqual(window._cache_setting_values()["pacmap_neighbors"], 24)
        # SSN is a physics cache whatever method is remembered.
        choose_layout(window, "ssn")
        physics = window._cache_setting_values()
        self.assertFalse(physics["umap_mode"])
        self.assertEqual(physics["layout_method"], "pacmap")      # the method it remembers
        self.assertEqual(build_compatibility("a" * 64, "b" * 64, "alignment", **physics)["layout_mode"], "physics")

    # --- the two saved settings ------------------------------------------------------------------

    def test_each_choice_is_saved_as_the_two_settings(self):
        window = self.open()
        for choice in ("umap", "pacmap", "ssn"):
            with self.subTest(choice=choice):
                choose_layout(window, choice)
                mode, method = self.saved_settings(window)
                self.assertEqual(mode, choice != "ssn")
                if choice != "ssn":
                    self.assertEqual(method, choice)
        # Values are real booleans and the two names, as every reader of the file expects.
        data = window.collect_data()
        self.assertIs(data["UMAP_MODE"], False)
        self.assertIn(data["LAYOUT_METHOD"], ("umap", "pacmap"))

    def test_ssn_leaves_layout_method_as_it_was_saved(self):
        for saved_method in ("umap", "pacmap"):
            with self.subTest(saved=saved_method):
                window = self.open({"UMAP_MODE": False, "LAYOUT_METHOD": saved_method})
                self.assertEqual(window.cb_layout_method.currentData(), "ssn")
                self.assertEqual(self.saved_settings(window), (False, saved_method))
                # Going to a manifold method and back to SSN keeps what that method was.
                other = "umap" if saved_method == "pacmap" else "pacmap"
                choose_layout(window, other)
                self.assertEqual(self.saved_settings(window), (True, other))
                choose_layout(window, "ssn")
                self.assertEqual(self.saved_settings(window), (False, other))
                # And loading the saved pair with UMAP_MODE on shows that method again.
                defaults = dict(self.namespace["TAB_PROFILE_SPECS"]["inputs_outputs"]["defaults"])
                window._apply_profile_data("inputs_outputs", {
                    **defaults, "UMAP_MODE": True, "LAYOUT_METHOD": saved_method})
                self.assertEqual(window.cb_layout_method.currentData(), saved_method)

    def test_a_window_opened_with_each_saved_pair(self):
        # (UMAP_MODE, LAYOUT_METHOD) as saved -> what the dropdown shows.
        cases = (
            ({}, "ssn", (False, "umap")),
            ({"UMAP_MODE": False}, "ssn", (False, "umap")),
            ({"UMAP_MODE": False, "LAYOUT_METHOD": "pacmap"}, "ssn", (False, "pacmap")),
            ({"UMAP_MODE": True}, "umap", (True, "umap")),
            ({"UMAP_MODE": True, "LAYOUT_METHOD": "umap"}, "umap", (True, "umap")),
            ({"UMAP_MODE": True, "LAYOUT_METHOD": "pacmap"}, "pacmap", (True, "pacmap")),
            ({"UMAP_MODE": "true", "LAYOUT_METHOD": "pacmap"}, "pacmap", (True, "pacmap")),
        )
        for saved, shown, written in cases:
            with self.subTest(saved=saved):
                window = self.open(saved)
                self.assertEqual(window.cb_layout_method.currentData(), shown)
                self.assertEqual(window.uses_manifold_layout(), shown != "ssn")
                self.assertEqual(self.saved_settings(window), written)

    def test_a_window_opened_with_saved_pacmap_settings(self):
        window = self.open({"UMAP_MODE": True, "LAYOUT_METHOD": "pacmap", "PACMAP_NEIGHBORS": 12})
        self.assertEqual(window.cb_layout_method.currentData(), "pacmap")
        self.assertEqual(window.spin_pacmap_k.value(), 12)
        self.assertTrue(window.inputs["PACMAP_NEIGHBORS"].isEnabled())
        self.assertFalse(window.inputs["UMAP_NEIGHBORS"].isEnabled())
        self.assertEqual(window._input_mode_stacks[1].currentWidget(), window.spin_pacmap_k)
        self.assertFalse(window.tabs.isTabVisible(2))

    def test_settings_saved_before_the_method_existed_open_as_umap(self):
        window = self.open({"UMAP_MODE": True, "UMAP_NEIGHBORS": 20, "UMAP_MIN_DIST": 0.3})
        self.assertEqual(window.cb_layout_method.currentData(), "umap")
        self.assertEqual(window.spin_pacmap_k.value(), 10)
        self.assertEqual((window.spin_umap_k.value(), window.spin_umap_md.value()), (20, 0.3))
        self.assertTrue(window.inputs["UMAP_NEIGHBORS"].isEnabled())
        self.assertEqual(window._input_mode_stacks[1].currentWidget(), window.spin_umap_k)

    def test_profiles_carry_the_two_settings_and_old_ones_default(self):
        window = self.open()
        defaults = dict(self.namespace["TAB_PROFILE_SPECS"]["inputs_outputs"]["defaults"])
        self.assertEqual((defaults["LAYOUT_METHOD"], defaults["PACMAP_NEIGHBORS"]), ("umap", 10))
        old = {key: value for key, value in defaults.items() if key not in ("LAYOUT_METHOD", "PACMAP_NEIGHBORS")}
        normalized = window._normalize_profile_data("inputs_outputs", old)
        self.assertEqual((normalized["LAYOUT_METHOD"], normalized["PACMAP_NEIGHBORS"]), ("umap", 10))
        window._apply_profile_data("inputs_outputs", {**defaults, "UMAP_MODE": True, "LAYOUT_METHOD": "pacmap",
                                                      "PACMAP_NEIGHBORS": 31})
        self.assertEqual(window.cb_layout_method.currentData(), "pacmap")
        self.assertEqual(window.spin_pacmap_k.value(), 31)
        saved = window._collect_tab_profile_data("inputs_outputs")
        self.assertEqual((saved["LAYOUT_METHOD"], saved["PACMAP_NEIGHBORS"], saved["UMAP_MODE"]), ("pacmap", "31", True))
        self.assertEqual(window._normalize_profile_data("inputs_outputs", saved)["PACMAP_NEIGHBORS"], 31)
        window._apply_profile_data("inputs_outputs", dict(defaults))
        self.assertEqual((window.cb_layout_method.currentData(), window.spin_pacmap_k.value()), ("ssn", 10))

    def test_a_profile_is_applied_whatever_order_its_two_settings_arrive_in(self):
        window = self.open()
        defaults = dict(self.namespace["TAB_PROFILE_SPECS"]["inputs_outputs"]["defaults"])
        for mode, method, shown in ((True, "pacmap", "pacmap"), (True, "umap", "umap"),
                                    (False, "pacmap", "ssn"), (False, "umap", "ssn")):
            for order in (("UMAP_MODE", "LAYOUT_METHOD"), ("LAYOUT_METHOD", "UMAP_MODE")):
                with self.subTest(mode=mode, method=method, order=order):
                    choose_layout(window, "pacmap" if method == "umap" else "umap")
                    values = {"UMAP_MODE": mode, "LAYOUT_METHOD": method}
                    for key in order:
                        window._set_widget_profile_value(key, values[key])
                    self.assertEqual(window.cb_layout_method.currentData(), shown)
                    self.assertEqual(self.saved_settings(window), (mode, method))
        # The (default) profile and a loaded profile follow the dropdown the same way.
        self.assertEqual(defaults["UMAP_MODE"], False)

    def test_a_profile_with_a_wrong_method_or_count_is_refused(self):
        window = self.open()
        defaults = dict(self.namespace["TAB_PROFILE_SPECS"]["inputs_outputs"]["defaults"])
        for changes, message in (({"LAYOUT_METHOD": "tsne"}, "invalid value for LAYOUT_METHOD"),
                                 ({"LAYOUT_METHOD": "ssn"}, "invalid value for LAYOUT_METHOD"),
                                 ({"PACMAP_NEIGHBORS": 1}, "invalid value for PACMAP_NEIGHBORS; expected 2 to 50"),
                                 ({"PACMAP_NEIGHBORS": 51}, "invalid value for PACMAP_NEIGHBORS; expected 2 to 50"),
                                 ({"PACMAP_NEIGHBORS": 2.5}, "expected an integer")):
            with self.subTest(changes=changes), self.assertRaises(ValueError) as caught:
                window._normalize_profile_data("inputs_outputs", {**defaults, **changes})
            self.assertIn(message, str(self.namespace["display_text"](caught.exception)))

    def test_a_language_redraw_carries_the_choice_and_the_remembered_method(self):
        for saved_method in ("umap", "pacmap"):
            for choice in ("ssn", "umap", "pacmap"):
                with self.subTest(saved=saved_method, choice=choice):
                    window = self.open({"UMAP_MODE": False, "LAYOUT_METHOD": saved_method})
                    if choice != "ssn":
                        choose_layout(window, choice)
                    carried = window.language_carry_over()
                    expected = (choice != "ssn", choice if choice != "ssn" else saved_method)
                    self.assertEqual((carried["values"]["UMAP_MODE"], carried["values"]["LAYOUT_METHOD"]), expected)
                    other = self.open()
                    other._restore_carried(carried)
                    self.assertEqual(other.cb_layout_method.currentData(), choice)
                    self.assertEqual(self.saved_settings(other), expected)

    def test_the_layout_settings_the_window_hands_to_the_generator(self):
        window = self.open()
        window.inputs["NODE_FASTA_FILE"].clear()
        window.inputs["NODE_FASTA_FILE"].addItem("set.fasta")
        window.inputs["INPUT_HDF5"].clear()
        window.inputs["INPUT_HDF5"].addItem("net.h5")

        def ready():
            # A choice re-resolves the cache folder, so the cache state is set after it.
            window._cache_launch_allowed = True
            window.current_cache_folder = str(Path(window.inputs["SAVED_LAYOUT_DIR"].text()) / "folder")
            window.cb_cache_file.clear()
            window.cb_cache_file.addItem("(New Layout Cache)", None)
            window.line_new_cache.setText("version_00")

        choose_layout(window, "pacmap")
        window.spin_pacmap_k.setValue(14)
        ready()
        settings = window._collect_layout_generation_settings()
        self.assertEqual((settings.UMAP_MODE, settings.LAYOUT_METHOD, settings.PACMAP_NEIGHBORS), (True, "pacmap", 14))
        self.assertTrue(settings.uses_pacmap())
        document = settings.to_document(project_root=Path(SRC).parent)
        self.assertEqual((document["layout"]["LAYOUT_METHOD"], document["layout"]["PACMAP_NEIGHBORS"]), ("pacmap", 14))
        # SSN hands over physics, with the method it remembers.
        choose_layout(window, "ssn")
        window.spin_thresh.setOptionalValue(0.5)      # a physics layout needs an edge filter
        ready()
        settings = window._collect_layout_generation_settings()
        self.assertEqual((settings.UMAP_MODE, settings.LAYOUT_METHOD), (False, "pacmap"))
        self.assertFalse(settings.uses_pacmap())

    # --- the tip ---------------------------------------------------------------------------------

    def test_the_tip_explains_all_three_choices(self):
        window = self.open()
        tips = window.tip_db_keys
        method = tips["LAYOUT_METHOD"]
        self.assertEqual([option.value for option in method.options], ["ssn", "umap", "pacmap"])
        self.assertNotIn("UMAP_MODE", tips)
        text = " ".join([method.summary, *method.details, *(option.text for option in method.options),
                         *(note.text for note in method.notes)])
        for words in ("SSN", "UMAP", "PaCMAP", "force-directed", "Simulation & Physics", "strongest edges"):
            self.assertIn(words, text)
        self.assertNotIn("Plot Manifold Instead", text)
        self.assertTrue(any("PaCMAP does not use it" in note.text for note in tips["UMAP_MIN_DIST"].notes))
        self.assertIn("2", tips["PACMAP_NEIGHBORS"].summary)
        self.assertIn("50", tips["PACMAP_NEIGHBORS"].summary)
        # The field the tip describes: the dropdown, with SSN as its default, and both its settings.
        from desktop.Tip_Panel import resolve
        entry = window.tip_db[window.cb_layout_method]
        self.assertIs(window.tip_db[window.labels["LAYOUT_METHOD"]], entry)
        facts = dict(resolve(entry).facts)
        self.assertEqual(facts["Default"], "SSN")
        self.assertEqual(facts["Setting"], "UMAP_MODE, LAYOUT_METHOD")

    def test_no_text_names_the_old_switch(self):
        window = self.open()
        for key, tip in window.tip_db_keys.items():
            body = " ".join([tip.summary, *tip.details, *(note.text for note in tip.notes),
                             *(option.text for option in tip.options)])
            self.assertNotIn("Plot Manifold", body, key)
        # The low resource tips name the control by its label.
        label = window.labels["LAYOUT_METHOD"].text().rstrip(":")
        for key in ("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_DRAWING", "LOW_RESOURCE_WEBSERVER"):
            body = " ".join(note.text for note in window.tip_db_keys[key].notes)
            self.assertIn(f"With {label} set to UMAP or PaCMAP", body, key)
        self.assertNotIn("Plot Manifold Instead", window.btn_clear_top_edge.toolTip())
        self.assertIn("Layout Method is SSN", window.btn_clear_top_edge.toolTip())


class ChineseLayoutMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utilities import Hardware_Acceleration  # noqa: F401 - torch before PySide6
        from desktop.Desktop_App import install_translations

        cls.namespace = load_config_namespace()
        cls.app = cls.namespace["QApplication"].instance() or cls.namespace["QApplication"]([])
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
        cls.app.processEvents()

    def test_the_label_is_translated_and_the_values_are_not(self):
        window = self.window
        self.assertEqual(window.labels["LAYOUT_METHOD"].text(), "布局方法：")
        self.assertTrue(any("一" <= character <= "鿿" for character in window.labels["PACMAP_NEIGHBORS"].text()))
        combo = window.cb_layout_method
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["SSN", "UMAP", "PaCMAP"])
        self.assertEqual([combo.itemData(i) for i in range(combo.count())], ["ssn", "umap", "pacmap"])
        choose_layout(window, "pacmap")
        self.assertEqual(window._widget_profile_value("LAYOUT_METHOD"), "pacmap")
        self.assertIs(window._widget_profile_value("UMAP_MODE"), True)
        choose_layout(window, "ssn")
        self.assertIs(window._widget_profile_value("UMAP_MODE"), False)

    def test_the_tip_is_translated(self):
        method = self.window.tip_db_keys["LAYOUT_METHOD"]
        for text in (method.summary, *(option.text for option in method.options), *(note.text for note in method.notes)):
            self.assertTrue(any("一" <= character <= "鿿" for character in text), text)


if __name__ == "__main__":
    unittest.main()
