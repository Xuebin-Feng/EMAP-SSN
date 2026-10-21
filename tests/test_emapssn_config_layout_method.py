"""The Config's Layout Method: UMAP or PaCMAP for the manifold layout (Plot Manifold Instead).

On the Inputs tab the switch Plot Manifold Instead (UMAP_MODE) swaps the two edge-filter fields for the
manifold layout's own: UMAP's neighbour count and minimum distance, or, when Layout Method says PaCMAP,
PaCMAP's neighbour count with the minimum distance greyed out (PaCMAP has none). The Layout Method dropdown
sits under that row and is enabled with the switch. Saved settings, profiles and the cache identity the window
works out all carry the two new settings, and anything saved before them loads as UMAP.
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

from tests.config_gui_loader import directory_globals, directory_settings, load_config_namespace, open_config_window
from tests.theme_fixture import apply_theme_for_class


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

    def test_the_widgets_and_their_defaults(self):
        window = self.open()
        self.assertEqual(window.cb_layout_method.count(), 2)
        self.assertEqual([window.cb_layout_method.itemData(i) for i in range(2)], ["umap", "pacmap"])
        self.assertEqual([window.cb_layout_method.itemText(i) for i in range(2)], ["UMAP", "PaCMAP"])
        self.assertEqual(window.cb_layout_method.currentData(), "umap")
        self.assertEqual((window.spin_pacmap_k.minimum(), window.spin_pacmap_k.maximum(), window.spin_pacmap_k.value()),
                         (2, 50, 10))
        self.assertEqual(window.labels["UMAP_MODE"].text(), "Plot Manifold Instead:")
        self.assertEqual(window.labels["LAYOUT_METHOD"].text(), "Layout Method:")
        self.assertEqual(window.labels["PACMAP_NEIGHBORS"].text(), "PaCMAP Nearest Neighbors:")
        for key in ("LAYOUT_METHOD", "PACMAP_NEIGHBORS", "UMAP_MODE"):
            self.assertIn(key, window.inputs)
            self.assertIn(key, window.tip_db_keys)

    def test_each_mode_shows_its_own_fields(self):
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
        window.check_umap.setChecked(True)
        self.assertEqual(shown(), ("UMAP Nearest Neighbors:", "UMAP Min Distance:", False, True))
        window.cb_layout_method.setCurrentIndex(window.cb_layout_method.findData("pacmap"))
        self.assertEqual(shown(), ("PaCMAP Nearest Neighbors:", "UMAP Min Distance:", True, False))
        window.cb_layout_method.setCurrentIndex(window.cb_layout_method.findData("umap"))
        self.assertEqual(shown()[:2], ("UMAP Nearest Neighbors:", "UMAP Min Distance:"))
        window.check_umap.setChecked(False)
        window.cb_layout_method.setCurrentIndex(window.cb_layout_method.findData("pacmap"))
        # Physics ignores the method: the filters show, and the manifold fields are off.
        self.assertEqual(shown()[:2], ("Similarity Threshold:", "Top Edge %:"))

    def test_what_is_enabled(self):
        window = self.open()
        keys = ("UMAP_NEIGHBORS", "UMAP_MIN_DIST", "PACMAP_NEIGHBORS", "LAYOUT_METHOD")

        def enabled():
            return {key: (window.inputs[key].isEnabled(), window.labels[key].isEnabled()) for key in keys}

        both = lambda value: (value, value)       # noqa: E731
        window.check_umap.setChecked(False)
        self.assertEqual(enabled(), {key: both(False) for key in keys})
        window.check_umap.setChecked(True)
        self.assertEqual(enabled(), {"UMAP_NEIGHBORS": both(True), "UMAP_MIN_DIST": both(True),
                                     "PACMAP_NEIGHBORS": both(False), "LAYOUT_METHOD": both(True)})
        window.cb_layout_method.setCurrentIndex(window.cb_layout_method.findData("pacmap"))
        # PaCMAP has no minimum distance: its field and label stay, greyed out.
        self.assertEqual(enabled(), {"UMAP_NEIGHBORS": both(False), "UMAP_MIN_DIST": both(False),
                                     "PACMAP_NEIGHBORS": both(True), "LAYOUT_METHOD": both(True)})
        self.assertTrue(window.inputs["UMAP_MIN_DIST"].isVisibleTo(window))
        # The physics tab is hidden for either method, as before.
        self.assertFalse(window.tabs.isTabVisible(2))
        window.check_umap.setChecked(False)
        self.assertTrue(window.tabs.isTabVisible(2))

    def test_the_settings_the_window_collects(self):
        window = self.open()
        window.check_umap.setChecked(True)
        window.cb_layout_method.setCurrentIndex(window.cb_layout_method.findData("pacmap"))
        window.spin_pacmap_k.setValue(23)
        data = window.collect_data()
        self.assertEqual((data["LAYOUT_METHOD"], data["PACMAP_NEIGHBORS"], data["UMAP_MODE"]), ("pacmap", "23", True))
        self.assertEqual(window._widget_profile_value("LAYOUT_METHOD"), "pacmap")
        self.assertEqual(window._widget_profile_value("PACMAP_NEIGHBORS"), "23")
        values = window._cache_setting_values()
        self.assertEqual((values["layout_method"], values["pacmap_neighbors"], values["umap_mode"]), ("pacmap", 23, True))
        # The cache identity the window works out is the one a layout job writes.
        from Cache_Manifest import build_compatibility
        compatibility = build_compatibility("a" * 64, "b" * 64, "alignment", **values)
        self.assertEqual(compatibility["layout_mode"], "pacmap")
        self.assertEqual(compatibility["edge_filter"], {"mode": "pacmap_neighbors", "value": 23})
        window.spin_pacmap_k.setValue(24)
        self.assertEqual(window._cache_setting_values()["pacmap_neighbors"], 24)

    def test_a_window_opened_with_saved_pacmap_settings(self):
        window = self.open({"UMAP_MODE": True, "LAYOUT_METHOD": "pacmap", "PACMAP_NEIGHBORS": 12})
        self.assertTrue(window.check_umap.isChecked())
        self.assertEqual(window.cb_layout_method.currentData(), "pacmap")
        self.assertEqual(window.spin_pacmap_k.value(), 12)
        self.assertTrue(window.inputs["PACMAP_NEIGHBORS"].isEnabled())
        self.assertFalse(window.inputs["UMAP_NEIGHBORS"].isEnabled())
        self.assertEqual(window._input_mode_stacks[1].currentWidget(), window.spin_pacmap_k)

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
        self.assertEqual((window.cb_layout_method.currentData(), window.spin_pacmap_k.value()), ("umap", 10))

    def test_a_profile_with_a_wrong_method_or_count_is_refused(self):
        window = self.open()
        defaults = dict(self.namespace["TAB_PROFILE_SPECS"]["inputs_outputs"]["defaults"])
        for changes, message in (({"LAYOUT_METHOD": "tsne"}, "invalid value for LAYOUT_METHOD"),
                                 ({"PACMAP_NEIGHBORS": 1}, "invalid value for PACMAP_NEIGHBORS; expected 2 to 50"),
                                 ({"PACMAP_NEIGHBORS": 51}, "invalid value for PACMAP_NEIGHBORS; expected 2 to 50"),
                                 ({"PACMAP_NEIGHBORS": 2.5}, "expected an integer")):
            with self.subTest(changes=changes), self.assertRaises(ValueError) as caught:
                window._normalize_profile_data("inputs_outputs", {**defaults, **changes})
            self.assertIn(message, str(self.namespace["display_text"](caught.exception)))

    def test_the_layout_settings_the_window_hands_to_the_generator(self):
        window = self.open()
        window.check_umap.setChecked(True)
        window.cb_layout_method.setCurrentIndex(window.cb_layout_method.findData("pacmap"))
        window.spin_pacmap_k.setValue(14)
        window.inputs["NODE_FASTA_FILE"].clear()
        window.inputs["NODE_FASTA_FILE"].addItem("set.fasta")
        window.inputs["INPUT_HDF5"].clear()
        window.inputs["INPUT_HDF5"].addItem("net.h5")
        window._cache_launch_allowed = True
        window.current_cache_folder = str(Path(window.inputs["SAVED_LAYOUT_DIR"].text()) / "folder")
        window.cb_cache_file.clear()
        window.cb_cache_file.addItem("(New Layout Cache)", None)
        window.line_new_cache.setText("version_00")
        settings = window._collect_layout_generation_settings()
        self.assertEqual((settings.UMAP_MODE, settings.LAYOUT_METHOD, settings.PACMAP_NEIGHBORS), (True, "pacmap", 14))
        self.assertTrue(settings.uses_pacmap())
        document = settings.to_document(project_root=Path(SRC).parent)
        self.assertEqual((document["layout"]["LAYOUT_METHOD"], document["layout"]["PACMAP_NEIGHBORS"]), ("pacmap", 14))

    def test_the_tips(self):
        window = self.open()
        tips = window.tip_db_keys
        method = tips["LAYOUT_METHOD"]
        self.assertEqual([option.value for option in method.options], ["umap", "pacmap"])
        text = " ".join([tips["UMAP_MODE"].summary, *tips["UMAP_MODE"].details])
        self.assertIn("UMAP or PaCMAP", text)
        self.assertTrue(any("PaCMAP does not use it" in note.text for note in tips["UMAP_MIN_DIST"].notes))
        self.assertIn("2", tips["PACMAP_NEIGHBORS"].summary)
        self.assertIn("50", tips["PACMAP_NEIGHBORS"].summary)
        # The low resource tips name the switch by its label.
        label = window.labels["UMAP_MODE"].text().rstrip(":")
        for key in ("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_DRAWING", "LOW_RESOURCE_WEBSERVER"):
            body = " ".join(note.text for note in tips[key].notes)
            self.assertIn(label, body, key)
            self.assertNotIn("Plot UMAP Instead", body)


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

    def test_the_new_texts_are_translated_and_the_values_are_not(self):
        window = self.window
        for key in ("UMAP_MODE", "LAYOUT_METHOD", "PACMAP_NEIGHBORS"):
            shown = window.labels[key].text()
            self.assertTrue(any("一" <= character <= "鿿" for character in shown), key)
        self.assertEqual([window.cb_layout_method.itemText(i) for i in range(2)], ["UMAP", "PaCMAP"])
        self.assertEqual([window.cb_layout_method.itemData(i) for i in range(2)], ["umap", "pacmap"])
        window.check_umap.setChecked(True)
        window.cb_layout_method.setCurrentIndex(1)
        self.assertEqual(window._widget_profile_value("LAYOUT_METHOD"), "pacmap")
        window.check_umap.setChecked(False)


if __name__ == "__main__":
    unittest.main()
