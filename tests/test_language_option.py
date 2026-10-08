# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Language option: the setting, the dropdown, and redrawing a window in another language.

One LANGUAGE setting in app_settings.json serves every window. Config and
Tools each show a Language dropdown. Choosing a language saves it and
redraws that window in it at once, keeping everything the window shows: its
size, position and splitters, its tabs and scroll positions, and every
field, unsaved edits included. Other windows take the language when they
next open. No real language ships yet, so the redraws here go to the
test-only pseudo-language, and a small compiled catalog stands in for a
real one. Simplified Chinese also brings its bundled font, only while it
shows.
"""

import ast
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
for folder in (SRC, SRC / "resources" / "languages"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QTextLayout
from PySide6.QtWidgets import QAbstractScrollArea, QApplication, QSplitter, QTabWidget, QTextBrowser

from desktop import Desktop_App
from desktop.Desktop_App import (
    ENGLISH,
    PSEUDO_LANGUAGE,
    QT_MONOSPACE_FAMILIES,
    QT_SIMPLIFIED_CHINESE_FAMILY,
    QT_UI_FAMILIES,
    SYSTEM_LANGUAGE,
    LanguageSelector,
    catalog_languages,
    configured_language,
    install_translations,
    installed_language,
    language_name,
    redraw_in_language,
    resolve_language,
    startup_language,
    system_language,
)
from utilities.App_Settings import read_app_settings
from utilities.Localization import is_pseudo_translated
import Update_Translations
from tests.translation_fixtures import visible_texts

GERMAN = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="de" sourcelanguage="en">
<context>
    <name>LanguageSelector</name>
    <message>
        <source>System default ({language})</source>
        <translation>Wie das System ({language})</translation>
    </message>
</context>
</TS>
"""


def flush(app):
    for _ in range(4):
        app.processEvents()


@contextlib.contextmanager
def runtime_warning(test):
    """test.assertWarns(RuntimeWarning), without assertWarns.

    assertWarns reads __warningregistry__ on every loaded module, and a lazy
    transformers module answers that by importing what it may lack.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    test.assertTrue(any(issubclass(warning.category, RuntimeWarning) for warning in caught), "no RuntimeWarning")


def view(window):
    """What a redraw must keep of how a window looks."""
    return {
        "geometry": window.geometry().getRect(),
        "maximized": bool(window.windowState() & Qt.WindowState.WindowMaximized),
        "splitters": [splitter.sizes() for splitter in window.findChildren(QSplitter)],
        "tabs": [tabs.currentIndex() for tabs in window.findChildren(QTabWidget)],
        "scrolls": [
            (area.horizontalScrollBar().value(), area.verticalScrollBar().value())
            for area in window.findChildren(QAbstractScrollArea)
        ],
    }


class Controller:
    """Stands in for the single-instance controller, which raises the window on a second start."""

    def __init__(self):
        self.callback = None

    def set_activation_callback(self, callback):
        self.callback = callback


class LanguageTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        cls.catalogs = Path(folder.name)
        for name in ("emapssn_de.ts", "emapssn_zh_CN.ts", "emapssn_en.ts"):
            (cls.catalogs / name).write_text(GERMAN.replace('language="de"', ""), encoding="utf-8")
            Update_Translations.run_qt_tool("lrelease", [cls.catalogs / name, "-qm", cls.catalogs / name.replace(".ts", ".qm")])
        (cls.catalogs / "emapssn.ts").write_text("<TS/>", encoding="utf-8")
        (cls.catalogs / "notes.txt").write_text("not a catalog", encoding="utf-8")

    def setUp(self):
        # Every test starts English, from a settings file of its own.
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.settings = Path(folder.name) / "app_settings.json"
        patcher = mock.patch.dict(os.environ, {"SSN_APP_SETTINGS_PATH": str(self.settings)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(install_translations, self.app, None)


class LanguageSettingTests(LanguageTestCase):
    def test_the_languages_are_those_with_a_compiled_catalog(self):
        self.assertEqual(catalog_languages(self.catalogs), ["de", "zh_CN"])
        self.assertEqual(catalog_languages(Desktop_App.LANGUAGES_DIR), [])

    def test_each_language_is_named_in_itself(self):
        names = {code: language_name(code) for code in ("en", "de", "fr", "ja", "zh_CN", "zh_TW")}
        self.assertEqual(names, {
            "en": "English", "de": "Deutsch", "fr": "Français", "ja": "日本語",
            "zh_CN": "简体中文", "zh_TW": "繁體中文",
        })

    def test_the_system_language_is_the_first_one_preferred_that_has_a_catalog(self):
        cases = [
            (["zh-Hans-CN", "en-US"], ["zh_CN"], "zh_CN"),
            (["en-CA", "zh-CN"], ["zh_CN"], ENGLISH),
            (["fr-FR", "de-AT"], ["de"], "de"),
            (["fr-FR"], ["de"], ENGLISH),
            (["pt-BR"], ["pt_BR", "pt"], "pt_BR"),
        ]
        for preferred, available, expected in cases:
            with self.subTest(preferred=preferred):
                self.assertEqual(system_language(available, preferred), expected)

    def test_the_setting_follows_the_system_until_one_is_saved(self):
        self.assertEqual(configured_language(), SYSTEM_LANGUAGE)
        self.assertEqual(configured_language({"LANGUAGE": "de"}), "de")
        for odd in (5, "", None, ["de"]):
            self.assertEqual(configured_language({"LANGUAGE": odd}), SYSTEM_LANGUAGE)

    def test_an_unreadable_settings_file_counts_as_no_setting(self):
        self.settings.write_text("{not json", encoding="utf-8")
        with runtime_warning(self):
            self.assertEqual(configured_language(), SYSTEM_LANGUAGE)

    def test_a_setting_resolves_to_a_language_with_a_catalog_or_english(self):
        self.assertEqual(resolve_language("de", self.catalogs), "de")
        self.assertIsNone(resolve_language(ENGLISH, self.catalogs))
        self.assertEqual(resolve_language(SYSTEM_LANGUAGE, self.catalogs, ["zh-CN"]), "zh_CN")
        self.assertIsNone(resolve_language(SYSTEM_LANGUAGE, self.catalogs, ["fr-FR"]))
        with runtime_warning(self):
            self.assertIsNone(resolve_language("ko", self.catalogs))

    def test_windows_start_in_the_saved_language(self):
        self.settings.write_text(json.dumps({"LANGUAGE": "zh_CN"}), encoding="utf-8")
        self.assertEqual(startup_language({}, catalog_dir=self.catalogs), "zh_CN")
        self.assertEqual(
            startup_language({"SSN_PSEUDO_TRANSLATION": "1"}, catalog_dir=self.catalogs), PSEUDO_LANGUAGE
        )
        self.assertIsNone(startup_language({}), "no catalog ships yet, so the program shows English")


class LanguageSelectorTests(LanguageTestCase):
    def make(self, **options):
        options.setdefault("catalog_dir", self.catalogs)
        options.setdefault("ui_languages", ["de-DE"])
        selector = LanguageSelector(**options)
        self.addCleanup(selector.deleteLater)
        return selector

    def test_it_lists_the_system_language_english_and_every_catalog(self):
        selector = self.make()
        self.assertEqual(
            [(selector.itemData(i), selector.itemText(i)) for i in range(selector.count())],
            [(SYSTEM_LANGUAGE, "System default (Deutsch)"), (ENGLISH, "English"),
             ("de", "Deutsch"), ("zh_CN", "简体中文")],
        )
        self.assertTrue(selector.toolTip())

    def test_it_starts_at_the_saved_setting(self):
        self.settings.write_text(json.dumps({"LANGUAGE": "zh_CN"}), encoding="utf-8")
        self.assertEqual(self.make().setting(), "zh_CN")
        self.assertEqual(self.make(setting="de").setting(), "de")
        self.assertEqual(self.make(setting="gone").setting(), SYSTEM_LANGUAGE)

    def test_choosing_announces_the_new_setting_and_revert_goes_back(self):
        selector = self.make(setting="de")
        chosen = []
        selector.language_chosen.connect(chosen.append)
        selector.setCurrentIndex(selector.findData("zh_CN"))
        self.assertEqual(chosen, ["zh_CN"])
        selector.revert()
        self.assertEqual(selector.setting(), "de")
        selector.show_setting(ENGLISH)
        self.assertEqual(chosen, ["zh_CN"], "show_setting does not announce")
        selector.setCurrentIndex(selector.findData("de"))
        selector.revert()
        self.assertEqual(selector.setting(), ENGLISH, "show_setting makes the shown choice the saved one")

    def test_its_own_text_is_translated_but_the_language_names_are_not(self):
        install_translations(self.app, "de", catalog_dir=self.catalogs)
        selector = self.make()
        self.assertEqual(selector.itemText(0), "Wie das System (Deutsch)")
        self.assertEqual([selector.itemText(i) for i in range(1, 4)], ["English", "Deutsch", "简体中文"])


class LanguageFontTests(LanguageTestCase):
    """Simplified Chinese brings its bundled font while it shows, and only then.

    Other languages keep the system's fonts for the same characters, as
    Japanese does for kanji.
    """

    def drawn_in(self, families):
        """The families Qt draws 中文 in, from a font that lists families.

        Only for text the bundled font draws: offscreen on Windows,
        QRawFont.familyName() crashes on a system font's raw font.
        """
        font = QFont()
        font.setFamilies(list(families))
        layout = QTextLayout("中文", font)
        layout.beginLayout()
        layout.createLine()
        layout.endLayout()
        return {run.rawFont().familyName() for run in layout.glyphRuns()}

    def test_simplified_chinese_brings_its_font_and_takes_it_away_again(self):
        if QT_SIMPLIFIED_CHINESE_FAMILY in QFontDatabase.families():
            self.skipTest(f"{QT_SIMPLIFIED_CHINESE_FAMILY} is installed on this system")
        # Both stacks name it after the core family, where Qt finds it once registered.
        for families in (QT_UI_FAMILIES, QT_MONOSPACE_FAMILIES):
            self.assertEqual(families[1], QT_SIMPLIFIED_CHINESE_FAMILY)

        install_translations(self.app, "zh_CN", self.catalogs)
        self.assertEqual(set(QFontDatabase.styles(QT_SIMPLIFIED_CHINESE_FAMILY)), {"Regular", "Bold"})
        for families in (QT_UI_FAMILIES, QT_MONOSPACE_FAMILIES):
            self.assertEqual(self.drawn_in(families), {QT_SIMPLIFIED_CHINESE_FAMILY})

        for language in ("de", PSEUDO_LANGUAGE, None):
            with self.subTest(language=language):
                install_translations(self.app, "zh_CN", self.catalogs)
                install_translations(self.app, language, self.catalogs)
                self.assertNotIn(QT_SIMPLIFIED_CHINESE_FAMILY, QFontDatabase.families())

    def test_a_missing_font_file_leaves_the_language_with_the_system_fonts(self):
        missing = Desktop_App.LanguageFont(
            QT_SIMPLIFIED_CHINESE_FAMILY, "Missing", ("noto/Missing/Missing-Regular.ttf", "noto/Missing/Missing-Bold.ttf")
        )
        with mock.patch.dict(Desktop_App.LANGUAGE_FONTS, {"zh_CN": missing}), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            installed = install_translations(self.app, "zh_CN", self.catalogs)
        self.assertEqual(installed_language(), "zh_CN")
        self.assertEqual(installed.font_ids, ())
        self.assertTrue(any("could not be registered" in str(warning.message) for warning in caught))


class ConfigLanguageTests(LanguageTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from tests.config_gui_loader import load_config_namespace

        cls.namespace = load_config_namespace()

    def open_window(self):
        from tests.config_gui_loader import open_config_window

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        window = open_config_window(self.namespace["ConfigGUI"], folder.name)
        self.track(window)
        window.show()
        flush(self.app)
        return window

    def track(self, window):
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)

    def change_everything(self, window):
        """Change what a user can change, leaving it all unsaved."""
        select = self.namespace["select_combo_value"]
        window.resize(1234, 777)
        window.move(40, 60)
        window.main_split.setSizes([700, 500])
        window.left_split.setSizes([500, 220])
        select(window.profile_selectors["visual_effects"], "(default)")
        select(window.profile_selectors["simulation_physics"], "(new)")
        window.profile_name_inputs["simulation_physics"].setText("Faster settle")
        window.inputs["MAX_STEPS"].setText("4321")
        window.inputs["ALIGNMENT_REFERENCE"].setText("REF_7")
        window.inputs["SIMILARITY_THRESHOLD"].setOptionalValue(0.42)
        window.stat_display.setPlainText("Statistics computed before the switch.")
        window.tabs.setCurrentIndex(2)
        flush(self.app)
        scroll_bar = window.tabs.currentWidget().verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum() // 2)
        self.assertGreater(scroll_bar.value(), 0)

    def test_the_dropdown_sits_under_the_statistics_report(self):
        window = self.open_window()
        selector = window.language_selector
        self.assertIs(selector.window(), window)
        self.assertTrue(window.right_panel.isAncestorOf(selector))
        self.assertGreater(selector.mapTo(window.right_panel, selector.rect().topLeft()).y(),
                           window.stat_display.geometry().bottom())
        self.assertEqual(selector.setting(), SYSTEM_LANGUAGE)

    def test_a_redraw_keeps_everything_the_window_shows(self):
        window = self.open_window()
        controller = window.single_instance = Controller()
        self.change_everything(window)
        # Hashes already computed come along, so the new window needn't hash again.
        window._cache_hash_cache["input.fasta|1|2"] = {"sha256": "c" * 64}
        before_view, before = view(window), window.language_carry_over()
        scroll_before = window.tabs.currentWidget().verticalScrollBar().value()

        replacement = window.switch_language(PSEUDO_LANGUAGE)
        self.track(replacement)
        flush(self.app)

        self.assertIsNot(replacement, window)
        self.assertFalse(window.isVisible())
        self.assertTrue(replacement.isVisible())
        self.assertEqual(installed_language(), PSEUDO_LANGUAGE)
        self.assertEqual(view(replacement), before_view)
        self.assertEqual(replacement.tabs.currentWidget().verticalScrollBar().value(), scroll_before)
        after = replacement.language_carry_over()
        self.assertEqual(after["values"], before["values"])
        self.assertEqual(after["profiles"], before["profiles"])
        # Taken from the window, never read again from viewer_settings.json.
        self.assertEqual(after["custom_settings"], before["custom_settings"])
        self.assertEqual(after["cache_hashes"], before["cache_hashes"])
        self.assertFalse(replacement.profile_content_widgets["visual_effects"].isEnabled(), "(default) is read-only")
        self.assertTrue(replacement.profile_name_inputs["simulation_physics"].isVisible())
        self.assertIn("Statistics computed before the switch.", replacement.stat_display.toPlainText())
        self.assertTrue(is_pseudo_translated(replacement.language_selector.itemText(0)))
        with mock.patch.object(Desktop_App, "show_window_in_front") as shown:
            controller.callback()
        shown.assert_called_once_with(replacement)

        # The same text as a window built fresh in the language. (Showing a
        # window moves fields between containers, so only the order differs.)
        fresh = self.namespace["ConfigGUI"](carried=replacement.language_carry_over())
        self.track(fresh)
        self.assertEqual(sorted(visible_texts(replacement)), sorted(visible_texts(fresh)))

        back = replacement.switch_language(None)
        self.track(back)
        flush(self.app)
        self.assertEqual(view(back), before_view)
        self.assertEqual(back.language_carry_over()["values"], before["values"])
        self.assertEqual(back.language_selector.itemText(0), "System default (English)")

    def test_the_scoring_a_blast_network_hides_comes_back_after_a_redraw(self):
        window = self.open_window()
        select, value = self.namespace["select_combo_value"], self.namespace["combo_value"]
        select(window.cb_score_mode, "local")
        window.update_norm_mode_options()
        select(window.cb_norm_mode, "average_sequence")
        window._set_network_type_controls("blast")
        self.assertEqual(window.cb_score_mode.currentIndex(), -1)

        replacement = window.switch_language(PSEUDO_LANGUAGE)
        self.track(replacement)
        # The BLAST network blanks both modes again; leaving it shows the choice.
        replacement._set_network_type_controls("blast")
        replacement._set_network_type_controls("alignment")
        self.assertEqual(
            (value(replacement.cb_score_mode), value(replacement.cb_norm_mode)),
            ("local", "average_sequence"),
        )

    def test_a_maximized_window_stays_maximized(self):
        window = self.open_window()
        window.showMaximized()
        flush(self.app)
        replacement = window.switch_language(PSEUDO_LANGUAGE)
        self.track(replacement)
        self.assertTrue(replacement.windowState() & Qt.WindowState.WindowMaximized)

    def test_choosing_a_language_saves_it_for_every_window_and_redraws(self):
        window = self.open_window()
        selector = window.language_selector
        with mock.patch.object(Desktop_App, "resolve_language", lambda setting: PSEUDO_LANGUAGE):
            selector.setCurrentIndex(selector.findData(ENGLISH))
        self.assertEqual(read_app_settings(), {"LANGUAGE": ENGLISH})
        self.assertFalse(window.isVisible())
        self.assertEqual(installed_language(), PSEUDO_LANGUAGE)
        replacement = Desktop_App._redrawn_windows[-1]
        self.track(replacement)
        self.assertEqual(replacement.language_selector.setting(), ENGLISH)

    def test_the_same_language_needs_no_redraw(self):
        window = self.open_window()
        self.assertIs(window.switch_language(None), window)
        window.language_selector.setCurrentIndex(window.language_selector.findData(ENGLISH))
        self.assertTrue(window.isVisible())
        self.assertEqual(read_app_settings(), {"LANGUAGE": ENGLISH})

    def test_a_language_that_cannot_be_saved_changes_nothing(self):
        window = self.open_window()
        self.settings.write_text("{not json", encoding="utf-8")
        selector = window.language_selector
        with mock.patch.object(self.namespace["QMessageBox"], "critical") as critical:
            selector.setCurrentIndex(selector.findData(ENGLISH))
        critical.assert_called_once()
        self.assertEqual(selector.setting(), SYSTEM_LANGUAGE)
        self.assertTrue(window.isVisible())
        self.assertIsNone(installed_language())
        self.assertEqual(self.settings.read_text(encoding="utf-8"), "{not json")

    def test_a_redraw_that_fails_keeps_the_window_and_its_language(self):
        window = self.open_window()
        install_translations(self.app, PSEUDO_LANGUAGE)

        def broken():
            raise RuntimeError("cannot build")

        with self.assertRaisesRegex(RuntimeError, "cannot build"):
            redraw_in_language(window, None, broken)
        self.assertTrue(window.isVisible())
        self.assertEqual(installed_language(), PSEUDO_LANGUAGE)

    def test_the_cache_chosen_before_the_redraw_is_chosen_again_once_discovery_lists_it(self):
        window = self.open_window()
        cache_manifest = self.namespace["cache_manifest"]
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        layouts = root / "layouts"
        layouts.mkdir()
        for age, name in enumerate(("newer.h5", "older.h5")):
            (layouts / name).write_bytes(b"")
            os.utime(layouts / name, (1_700_000_000 - age, 1_700_000_000 - age))
        directory_input = window.inputs["SAVED_LAYOUT_DIR"]
        directory_input.blockSignals(True)
        directory_input.setText(str(root))
        directory_input.blockSignals(False)
        older = cache_manifest.relative_cache_path(str(root), str(layouts), "older.h5")

        def discover():
            records = {"sequence": {"sha256": "a" * 64}, "network": {"sha256": "b" * 64}, "network_type": "alignment"}
            with mock.patch.object(
                window, "_cache_paths_from_inputs", return_value=(str(root / "set.fasta"), str(root / "network.h5"))
            ), mock.patch.object(
                cache_manifest, "build_canonical_cache_name", return_value="canonical"
            ), mock.patch.object(
                cache_manifest, "find_matching_manifest_folders", return_value=[{"folder": str(layouts)}]
            ):
                window._apply_cache_discovery(records)

        discover()
        self.assertEqual(window.cb_cache_file.currentText(), "newer.h5", window.lbl_cache_tracker.text())
        for choice, shown in (
            ({"new": False, "data": older, "name": ""}, ("older.h5", "")),
            ({"new": True, "data": None, "name": "my_layout"}, ("(New Layout Cache)", "my_layout")),
        ):
            with self.subTest(choice=choice):
                window._carried_cache_choice = choice
                discover()
                self.assertEqual((window.cb_cache_file.currentText(), window.line_new_cache.text()), shown)
                self.assertIsNone(window._carried_cache_choice, "applied once")


class ToolsLanguageTests(LanguageTestCase):
    def open_window(self):
        import EMAPSSN_Tools
        from tests.tools_gui_fixtures import isolated_tools_project

        self.project = isolated_tools_project(self)
        patcher = mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser)
        patcher.start()
        self.addCleanup(patcher.stop)
        window = EMAPSSN_Tools.ToolsGUI()
        self.track(window)
        window.show()
        flush(self.app)
        return window

    def track(self, window):
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)

    def test_the_dropdown_sits_under_the_script_description(self):
        window = self.open_window()
        selector = window.language_selector
        self.assertTrue(window.right_widget.isAncestorOf(selector))
        self.assertGreater(selector.mapTo(window.right_widget, selector.rect().topLeft()).y(),
                           window.script_desc_text.geometry().bottom())

    def test_a_redraw_keeps_everything_the_window_shows_and_never_rereads_the_file(self):
        window = self.open_window()
        window.single_instance = Controller()
        window.resize(1300, 820)
        window.move(30, 50)
        window.splitter.setSizes([900, 400])
        window.tabs.setCurrentIndex(1)
        edited = 0
        for data in window.script_data.values():
            for entry in data["inputs"].values():
                if entry["type"] == "number" and edited < 3:
                    entry["widget"].setValue(entry["widget"].value() + 1)
                    edited += 1
                elif entry["type"] == "switch" and edited < 6:
                    entry["widget"].toggle()
                    edited += 1
        self.assertEqual(edited, 6)
        key = next(iter(window.dir_inputs))
        window.dir_inputs[key].setText(window.dir_inputs[key].text() + "_unsaved")
        flush(self.app)
        area = window.tabs.currentWidget()
        area.verticalScrollBar().setValue(area.verticalScrollBar().maximum() // 2)
        before_view, before = view(window), window.language_carry_over()
        scroll_before = area.verticalScrollBar().value()
        # A redraw shows the window's values, not what the file holds.
        (self.project / "tools_settings.json").write_text(json.dumps({"DIRECTORIES": {}}), encoding="utf-8")

        replacement = window.switch_language(PSEUDO_LANGUAGE)
        self.track(replacement)
        flush(self.app)

        self.assertFalse(window.isVisible())
        self.assertEqual(view(replacement), before_view)
        self.assertEqual(replacement.tabs.currentWidget().verticalScrollBar().value(), scroll_before)
        self.assertEqual(replacement.language_carry_over()["document"], before["document"])
        self.assertTrue(is_pseudo_translated(replacement.language_selector.itemText(0)))

        back = replacement.switch_language(None)
        self.track(back)
        self.assertEqual(back.language_carry_over()["document"], before["document"])
        self.assertEqual(view(back), before_view)

    def test_a_language_that_cannot_be_saved_changes_nothing(self):
        import EMAPSSN_Tools

        window = self.open_window()
        self.settings.write_text("[]", encoding="utf-8")
        selector = window.language_selector
        with mock.patch.object(EMAPSSN_Tools.QMessageBox, "critical") as critical:
            selector.setCurrentIndex(selector.findData(ENGLISH))
        critical.assert_called_once()
        self.assertEqual(selector.setting(), SYSTEM_LANGUAGE)
        self.assertTrue(window.isVisible())


class StartupHandOffTests(unittest.TestCase):
    """The started window gives a redraw its single-instance controller to pass on."""

    def test_config_and_tools_hand_their_controller_to_the_window(self):
        for script, window_class in (("EMAPSSN_Config.py", "ConfigGUI"), ("EMAPSSN_Tools.py", "ToolsGUI")):
            with self.subTest(script=script):
                tree = ast.parse((SRC / script).read_text(encoding="utf-8"))
                handed = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Assign) and ast.unparse(node) == "window.single_instance = single_instance"
                ]
                built = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Assign) and ast.unparse(node) == f"window = {window_class}()"
                ]
                self.assertEqual(len(handed), 1)
                self.assertEqual(handed[0].lineno, built[0].lineno + 1)


if __name__ == "__main__":
    unittest.main()
